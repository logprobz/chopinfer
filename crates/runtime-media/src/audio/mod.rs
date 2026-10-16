//! Parameterized CPU audio preprocessing.
use crate::{DecodedAudio, MediaError, MediaLimits, MediaResult};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct AudioRecipe {
    pub sample_rate: u32,
    pub sinc_width: u32,
    pub rolloff: f64,
    pub n_fft: usize,
    pub window: usize,
    pub hop: usize,
    pub mel_bins: usize,
    pub f_min: f64,
    pub f_max: f64,
    pub log_floor: f32,
    pub first_kernel: usize,
    pub second_kernel: usize,
    pub second_stride: usize,
    pub pool: usize,
    pub group: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogMel {
    pub frames: usize,
    pub bins: usize,
    pub values: Vec<f32>,
    pub tokens: usize,
}

pub fn strided_group_tokens(frames: usize, recipe: AudioRecipe) -> MediaResult<usize> {
    if recipe.first_kernel != 3
        || recipe.second_kernel != 3
        || recipe.second_stride == 0
        || recipe.pool == 0
        || recipe.group == 0
    {
        return Err(MediaError::new("invalid audio token rule"));
    }
    let first = frames;
    let second = (first.saturating_sub(1) / recipe.second_stride) + 1;
    let pooled = second.div_ceil(recipe.pool);
    Ok(pooled.div_ceil(recipe.group))
}

pub fn resample(
    samples: &[f32],
    source_rate: u32,
    recipe: AudioRecipe,
    limits: MediaLimits,
) -> MediaResult<Vec<f32>> {
    resample_cancellable(samples, source_rate, recipe, limits, &|| false)
}

pub fn resample_cancellable(
    samples: &[f32],
    source_rate: u32,
    recipe: AudioRecipe,
    limits: MediaLimits,
    cancelled: &dyn Fn() -> bool,
) -> MediaResult<Vec<f32>> {
    check_cancelled(cancelled)?;
    if source_rate == 0
        || recipe.sample_rate == 0
        || recipe.sinc_width == 0
        || !(0.0..1.0).contains(&recipe.rolloff)
    {
        return Err(MediaError::new("invalid resample recipe"));
    }
    let count = (samples.len() as u64 * recipe.sample_rate as u64).div_ceil(source_rate as u64);
    limits.check_audio(count)?;
    for chunk in samples.chunks(4096) {
        check_cancelled(cancelled)?;
        if chunk.iter().any(|sample| !sample.is_finite()) {
            return Err(MediaError::new("nonfinite audio sample"));
        }
    }
    if source_rate == recipe.sample_rate {
        return Ok(samples.to_vec());
    }
    // torchaudio's Hann sinc kernel is sampled around each input-rate output position.
    let base = source_rate as f64 / recipe.sample_rate as f64;
    let width = recipe.sinc_width as f64 * base.max(1.0) / recipe.rolloff;
    let mut output = Vec::with_capacity(count as usize);
    for n in 0..count as usize {
        if n % 256 == 0 {
            check_cancelled(cancelled)?;
        }
        let center = n as f64 * base;
        let left = (center - width).floor() as isize;
        let right = (center + width).ceil() as isize;
        let mut sum = 0.0;
        for index in left..=right {
            // Bound work between polls even at unusually high source rates.
            if index & 4095 == 0 {
                check_cancelled(cancelled)?;
            }
            if !(0..samples.len() as isize).contains(&index) {
                continue;
            }
            let x = (index as f64 - center) / base.max(1.0) * recipe.rolloff;
            if x.abs() >= recipe.sinc_width as f64 {
                continue;
            }
            let sinc = if x == 0.0 {
                1.0
            } else {
                (PI * x).sin() / (PI * x)
            };
            let window = (1.0 + (PI * x / recipe.sinc_width as f64).cos()) * 0.5;
            sum += samples[index as usize] as f64 * sinc * window * recipe.rolloff / base.max(1.0);
        }
        let sample = sum as f32;
        if !sample.is_finite() {
            return Err(MediaError::new("nonfinite resampled audio"));
        }
        output.push(sample);
    }
    Ok(output)
}

#[derive(Clone, Copy, Default)]
struct Complex {
    re: f64,
    im: f64,
}
impl Complex {
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

#[cfg(test)]
fn fft(input: &[Complex]) -> MediaResult<Vec<Complex>> {
    let n = input.len();
    if n == 1 {
        return Ok(input.to_vec());
    }
    let radix = [2, 3, 5]
        .into_iter()
        .find(|r| n % r == 0)
        .ok_or_else(|| MediaError::new("FFT length needs 2, 3, 5 factors"))?;
    let sub = n / radix;
    let parts = (0..radix)
        .map(|r| fft(&(0..sub).map(|i| input[i * radix + r]).collect::<Vec<_>>()))
        .collect::<MediaResult<Vec<_>>>()?;
    let mut result = vec![Complex::default(); n];
    for k in 0..n {
        let mut value = Complex::default();
        for (r, part) in parts.iter().enumerate() {
            let phase = -2.0 * PI * (r * k) as f64 / n as f64;
            value = value.add(part[k % sub].mul(Complex {
                re: phase.cos(),
                im: phase.sin(),
            }));
        }
        result[k] = value;
    }
    Ok(result)
}

fn check_cancelled(cancelled: &dyn Fn() -> bool) -> MediaResult<()> {
    if cancelled() {
        Err(MediaError::new("audio preparation cancelled"))
    } else {
        Ok(())
    }
}

/// Cache twiddles without changing the recursive transform's multiplication
/// or addition order. Two scratch arrays serve every stage and every frame.
struct FftPlan {
    n: usize,
    radix: usize,
    twiddles: Vec<Complex>,
    child: Option<Box<FftPlan>>,
}
impl FftPlan {
    fn new(n: usize) -> MediaResult<Self> {
        if n == 1 {
            return Ok(Self {
                n,
                radix: 1,
                twiddles: Vec::new(),
                child: None,
            });
        }
        let radix = [2, 3, 5]
            .into_iter()
            .find(|r| n % r == 0)
            .ok_or_else(|| MediaError::new("FFT length needs 2, 3, 5 factors"))?;
        let child = Box::new(Self::new(n / radix)?);
        let twiddles = (0..n)
            .flat_map(|k| {
                (0..radix).map(move |r| {
                    let phase = -2.0 * PI * (r * k) as f64 / n as f64;
                    Complex {
                        re: phase.cos(),
                        im: phase.sin(),
                    }
                })
            })
            .collect();
        Ok(Self {
            n,
            radix,
            twiddles,
            child: Some(child),
        })
    }
    fn run(
        &self,
        input: &[Complex],
        stride: usize,
        output: &mut [Complex],
        scratch: &mut [Complex],
    ) {
        let Some(child) = &self.child else {
            output[0] = input[0];
            return;
        };
        let sub = child.n;
        for r in 0..self.radix {
            child.run(
                &input[r * stride..],
                stride * self.radix,
                &mut scratch[r * sub..(r + 1) * sub],
                &mut output[..sub],
            );
        }
        for (k, out) in output[..self.n].iter_mut().enumerate() {
            let mut value = Complex::default();
            for r in 0..self.radix {
                value =
                    value.add(scratch[r * sub + k % sub].mul(self.twiddles[k * self.radix + r]));
            }
            *out = value;
        }
    }
}

fn reflect(index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let period = 2 * (length - 1) as isize;
    let i = index.rem_euclid(period);
    if i < length as isize {
        i as usize
    } else {
        (period - i) as usize
    }
}
fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// torchaudio's `melscale_fbanks` (HTK scale, no norm), including its f32
/// band edges. The weights then enter the f64 accumulation unchanged.
fn mel_filters(recipe: AudioRecipe) -> Vec<Vec<f64>> {
    let linspace = |start: f64, end: f64, steps: usize| -> Vec<f32> {
        let (start, end) = (start as f32, end as f32);
        let step = (end - start) / (steps - 1) as f32;
        (0..steps)
            .map(|i| {
                if i < steps / 2 {
                    start + step * i as f32
                } else {
                    end - step * (steps - 1 - i) as f32
                }
            })
            .collect()
    };
    let bins = recipe.n_fft / 2 + 1;
    let freqs = linspace(0.0, (recipe.sample_rate / 2) as f64, bins);
    let points = linspace(
        hz_to_mel(recipe.f_min),
        hz_to_mel(recipe.f_max),
        recipe.mel_bins + 2,
    )
    .into_iter()
    .map(|mel| 700.0 * (10f32.powf(mel / 2595.0) - 1.0))
    .collect::<Vec<_>>();
    (0..recipe.mel_bins)
        .map(|m| {
            freqs
                .iter()
                .map(|&hz| {
                    let down = -(points[m] - hz) / (points[m + 1] - points[m]);
                    let up = (points[m + 2] - hz) / (points[m + 2] - points[m + 1]);
                    down.min(up).max(0.0) as f64
                })
                .collect()
        })
        .collect()
}

/// The frame counts [`log_mel`] emits: one per hop of a clip longer than
/// the reflect padding (`n_fft / 2` samples) and at most the limit's
/// samples, plus one.
pub fn emitted_frames(recipe: AudioRecipe, limits: MediaLimits) -> std::ops::RangeInclusive<u64> {
    let hop = recipe.hop.max(1) as u64;
    (recipe.n_fft as u64 / 2 + 1) / hop + 1..=limits.maximum_audio_frames / hop + 1
}

pub fn log_mel(
    audio: &DecodedAudio,
    recipe: AudioRecipe,
    limits: MediaLimits,
) -> MediaResult<LogMel> {
    log_mel_cancellable(audio, recipe, limits, &|| false)
}

pub fn log_mel_cancellable(
    audio: &DecodedAudio,
    recipe: AudioRecipe,
    limits: MediaLimits,
    cancelled: &dyn Fn() -> bool,
) -> MediaResult<LogMel> {
    check_cancelled(cancelled)?;
    if recipe.n_fft == 0
        || recipe.window == 0
        || recipe.window > recipe.n_fft
        || recipe.hop == 0
        || recipe.mel_bins == 0
        || recipe.sample_rate == 0
        || recipe.f_min < 0.0
        || recipe.f_max > recipe.sample_rate as f64 / 2.0
        || recipe.f_max <= recipe.f_min
        || recipe.log_floor <= 0.0
    {
        return Err(MediaError::new("invalid log-mel recipe"));
    }
    let samples =
        resample_cancellable(&audio.samples, audio.sample_rate, recipe, limits, cancelled)?;
    if samples.len() <= recipe.n_fft / 2 {
        return Err(MediaError::new("audio shorter than reflect padding"));
    }
    let frames = samples.len() / recipe.hop + 1;
    let tokens = strided_group_tokens(frames, recipe)?;
    limits.check_tokens(tokens as u64)?;
    let size = frames
        .checked_mul(recipe.mel_bins)
        .ok_or_else(|| MediaError::new("log-mel allocation overflow"))?;
    let filters: Vec<Vec<(usize, f64)>> = mel_filters(recipe)
        .into_iter()
        .map(|filter| {
            filter
                .into_iter()
                .enumerate()
                .filter(|(_, weight)| *weight != 0.0)
                .collect()
        })
        .collect();
    let plan = FftPlan::new(recipe.n_fft)?;
    let window: Vec<_> = (0..recipe.window)
        .map(|j| 0.5 - 0.5 * (2.0 * PI * j as f64 / recipe.window as f64).cos())
        .collect();
    let mut input = vec![Complex::default(); recipe.n_fft];
    let mut spectrum = vec![Complex::default(); recipe.n_fft];
    let mut scratch = vec![Complex::default(); recipe.n_fft];
    let mut magnitudes = vec![0.0; recipe.n_fft / 2 + 1];
    let mut values = Vec::with_capacity(size);
    let offset = (recipe.n_fft - recipe.window) / 2;
    for frame in 0..frames {
        check_cancelled(cancelled)?;
        for j in 0..recipe.window {
            let sample_index = frame as isize * recipe.hop as isize + j as isize + offset as isize
                - recipe.n_fft as isize / 2;
            input[j + offset].re = samples[reflect(sample_index, samples.len())] as f64 * window[j];
        }
        plan.run(&input, 1, &mut spectrum, &mut scratch);
        for (magnitude, z) in magnitudes.iter_mut().zip(&spectrum) {
            *magnitude = (z.re * z.re + z.im * z.im).sqrt();
        }
        for filter in &filters {
            let energy = filter
                .iter()
                .map(|(bin, weight)| weight * magnitudes[*bin])
                .sum::<f64>();
            let energy = energy as f32;
            // Check before max, which would silently replace NaN with the floor.
            if !energy.is_finite() {
                return Err(MediaError::new("nonfinite mel energy"));
            }
            let value = energy.max(recipe.log_floor).ln();
            if !value.is_finite() {
                return Err(MediaError::new("nonfinite log-mel value"));
            }
            values.push(value);
        }
    }
    Ok(LogMel {
        frames,
        bins: recipe.mel_bins,
        values,
        tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn recipe() -> AudioRecipe {
        AudioRecipe {
            sample_rate: 24_000,
            sinc_width: 6,
            rolloff: 0.99,
            n_fft: 960,
            window: 960,
            hop: 240,
            mel_bins: 128,
            f_min: 0.0,
            f_max: 12_000.0,
            log_floor: 1e-7,
            first_kernel: 3,
            second_kernel: 3,
            second_stride: 2,
            pool: 2,
            group: 4,
        }
    }
    fn limits() -> MediaLimits {
        MediaLimits {
            maximum_bytes: 1,
            maximum_pixels: 1,
            maximum_audio_frames: 400_000,
            maximum_frames: 1,
            maximum_tokens: 1000,
            maximum_aspect_ratio: 200.0,
        }
    }
    #[test]
    fn audio_preparation_observes_cancellation_during_work() {
        use std::cell::Cell;
        for rate in [24_000, 44_100, 48_000] {
            let polls = Cell::new(0);
            let audio = DecodedAudio {
                sample_rate: rate,
                samples: vec![0.25; rate as usize],
            };
            // Pass the finite-input scan, then interrupt resampling (or the
            // STFT when the input already has the target rate).
            let stop = 2 + (rate as usize).div_ceil(4096) + 8;
            let cancelled = || {
                polls.set(polls.get() + 1);
                polls.get() >= stop
            };
            let result = log_mel_cancellable(&audio, recipe(), limits(), &cancelled);
            assert!(result.is_err(), "{rate} Hz ignored cancellation");
            assert_eq!(polls.get(), stop);
        }
    }

    #[test]
    fn resampling_polls_inside_wide_source_windows() {
        let polls = std::cell::Cell::new(0);
        let result =
            resample_cancellable(&[0.25; 1024], 4_000_000_000, recipe(), limits(), &|| {
                polls.set(polls.get() + 1);
                polls.get() >= 4
            });
        assert!(result.is_err());
        assert_eq!(polls.get(), 4);
    }

    #[test]
    fn planned_fft_matches_recursive_bits() {
        for n in [1, 12, 30, 960] {
            let input: Vec<_> = (0..n)
                .map(|i| Complex {
                    re: (i as f64 * 0.7).sin(),
                    im: (i as f64 * 0.3).cos(),
                })
                .collect();
            let expected = fft(&input).unwrap();
            let mut output = vec![Complex::default(); n];
            FftPlan::new(n)
                .unwrap()
                .run(&input, 1, &mut output, &mut vec![Complex::default(); n]);
            for (a, b) in output.iter().zip(expected) {
                assert_eq!(
                    (a.re.to_bits(), a.im.to_bits()),
                    (b.re.to_bits(), b.im.to_bits())
                );
            }
        }
    }

    #[test]
    fn token_rule_matches_hand_count() {
        assert_eq!(strided_group_tokens(1595, recipe()).unwrap(), 100);
        assert_eq!(strided_group_tokens(1, recipe()).unwrap(), 1);
    }
    #[test]
    fn mixed_radix_fft_has_expected_impulse_and_constant() {
        for n in [12, 30, 960] {
            let mut impulse = vec![Complex::default(); n];
            impulse[0].re = 1.0;
            assert!(
                fft(&impulse)
                    .unwrap()
                    .iter()
                    .all(|z| (z.re - 1.0).abs() < 1e-10 && z.im.abs() < 1e-10)
            );
        }
    }
    #[test]
    fn silence_produces_log_floor_and_expected_frames() {
        let audio = DecodedAudio {
            sample_rate: 24_000,
            samples: vec![0.0; 960],
        };
        let spec = log_mel(&audio, recipe(), limits()).unwrap();
        assert_eq!(spec.frames, 5);
        assert_eq!(spec.bins, 128);
        assert!(spec.values.iter().all(|x| (*x - 1e-7f32.ln()).abs() < 1e-5));
    }

    #[test]
    fn emitted_frames_are_the_counts_log_mel_produces() {
        let limits = MediaLimits {
            maximum_audio_frames: 2_000,
            ..limits()
        };
        let frames = emitted_frames(recipe(), limits);
        assert_eq!(frames, 3..=9);
        let count = |samples: usize| {
            let audio = DecodedAudio {
                sample_rate: 24_000,
                samples: vec![0.0; samples],
            };
            log_mel(&audio, recipe(), limits).map(|mel| mel.frames as u64)
        };
        assert!(count(480).is_err());
        assert_eq!(count(481).unwrap(), *frames.start());
        assert_eq!(count(2_000).unwrap(), *frames.end());
        assert!(count(2_001).is_err());
        for samples in 481..=2_000 {
            assert!(frames.contains(&count(samples).unwrap()));
        }
    }

    #[test]
    fn resample_refuses_nonfinite_inputs_at_equal_and_different_rates() {
        for rate in [24_000, 12_000] {
            for sample in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
                assert!(resample(&[sample; 32], rate, recipe(), limits()).is_err());
            }
        }
    }

    #[test]
    fn resample_refuses_finite_samples_that_overflow_the_output() {
        assert!(resample(&[f32::MAX; 32], 12_000, recipe(), limits()).is_err());
    }

    #[test]
    fn log_mel_refuses_energy_that_overflows_float32() {
        let mut samples = vec![0.0; 960];
        samples[240] = f32::MAX;
        let audio = DecodedAudio {
            sample_rate: 24_000,
            samples,
        };
        assert!(log_mel(&audio, recipe(), limits()).is_err());
    }
}

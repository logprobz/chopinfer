#pragma once
#include <type_traits>

namespace chop::gemm::mxfp8_fold_policy {

// The trained E4M3 grid is already materialized at the entry boundary.
struct E4m3Ceil32 {};
struct Ue8m0AtK32 {};
struct FourK32 {};
struct AscendingF32 {};

// A new arithmetic policy needs its own numerical identity and qualification.
// Scheduling knobs cannot substitute another activation or reduction policy.
template<class Activation, class Scale, class Chain, class Reduction>
inline constexpr bool qualified =
    std::is_same_v<Activation, E4m3Ceil32> &&
    std::is_same_v<Scale, Ue8m0AtK32> &&
    std::is_same_v<Chain, FourK32> &&
    std::is_same_v<Reduction, AscendingF32>;

} // namespace chop::gemm::mxfp8_fold_policy

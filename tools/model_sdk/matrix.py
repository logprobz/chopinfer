"""Run four compatibility cells using two SDK pins and two prebuilt hosts."""
import argparse
import json
from pathlib import Path

from .source import git, sha
from .smoke import smoke
from .workflow import exercise, require, external_output
from .conformance import refusals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source','output','h0','h0-conformance','h1','h1-conformance']:
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ['s0','s1','h0-revision','h1-revision','host-features']:
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    external_output(source, output)
    require(not output.exists(), 'matrix output already exists')
    revisions = {label:git(source,'rev-parse','--verify',getattr(args,label)+'^{commit}').decode().strip()
                 for label in ['s0','s1','h0_revision','h1_revision']}
    require(revisions['s0'] != revisions['s1'], 'matrix needs two distinct SDK revisions')
    require(revisions['h0_revision'] != revisions['h1_revision'], 'matrix needs two distinct host revisions')
    hosts = {'h0':(args.h0.resolve(),args.h0_conformance.resolve()),
             'h1':(args.h1.resolve(),args.h1_conformance.resolve())}
    before = {str(p):sha(p) for pair in hosts.values() for p in pair}
    output.mkdir(parents=True)
    records = {}
    for sdk, host_label in [('s0','h0'),('s1','h1')]:
        host, conformance = hosts[host_label]
        records[f'{sdk}/{host_label}'] = smoke(source,output/sdk,host,conformance,
                                              revision=revisions[sdk],development=False,common=True)
    for sdk, host_label in [('s0','h1'),('s1','h0')]:
        host, conformance = hosts[host_label]
        root = output/f'{sdk}-{host_label}'
        root.mkdir()
        records[f'{sdk}/{host_label}'] = {
            'host':exercise(host,conformance,output/sdk/'package-1',root/'host'),
            'refusals':refusals(conformance,output/sdk/'package-1',output/sdk/'omitted-package',root/'refusals')}
    require(before == {p:sha(p) for p in before}, 'a prebuilt host changed')
    receipt = {'schema':'chopinfer-sdk-matrix-v1','revisions':revisions,'host_features':args.host_features,
               'host_sha256':before,'cells':records,
               'scope':'Independent-build mechanism evidence for these commits. V1 remains draft; no historical release or GPU qualification.'}
    (output/'matrix.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('All four independent SDK/host cells passed')


if __name__ == '__main__':
    main()

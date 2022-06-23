#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a json file with DWI, T1 and fmap informations from BIDS folder
"""

import os

import argparse
from bids import BIDSLayout
import json

from scilpy.io.utils import add_overwrite_arg, assert_outputs_exist


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    p.add_argument("in_bids",
                   help="Input BIDS folder.")

    p.add_argument("out_json",
                   help="Output json file.")

    p.add_argument('--participants_label', nargs="+",
                   help='The label(s) of the specific participant(s) you'
                        ' want to be be analyzed. Participants should not '
                        'include "sub-". If this parameter is not provided all'
                        ' subjects should be analyzed.')

    p.add_argument('--clean',
                   action='store_true',
                   help='If set, it will remove all the participants that '
                        'are missing any information.')

    p.add_argument("--readout", type=float, default=0.062,
                   help="Default total readout time value [%(default)s].")

    add_overwrite_arg(p)

    return p


def get_metadata(bf):
    """ Return the metadata of a BIDSFile

    Parameters
    ----------
    bf : BIDSFile object

    Returns
    -------
    Dictionnary containing the metadata
    """
    filename = bf.path.replace(
        '.' + bf.get_entities()['extension'], '')
    with open(filename + '.json', 'r') as handle:
        return json.load(handle)


def get_dwi_associations(fmaps, bvals, bvecs, sbrefs):
    """ Return DWI associations

    Parameters
    ----------
    fmaps : List of BIDSFile object
        List of field maps

    bvals : List of BIDSFile object
        List of b-value files

    bvecs : List of BIDSFile object
        List of b-vector files

    Returns
    -------
    Dictionnary containing the files associated to a DWI
    {dwi_filename: {'bval': bval_filename,
                    'bvec': bvec_filename,
                    'fmap': fmap_filename}}
    """
    associations = {}

    # Associate b-value files
    for bval in bvals:
        dwi_filename = os.path.basename(bval.path).replace('.bval', '.nii.gz')
        if dwi_filename not in associations.keys():
            associations[dwi_filename] = {"bval": bval.path}
        else:
            associations[dwi_filename]["bval"] = bval.path

    # Associate b-vector files
    for bvec in bvecs:
        dwi_filename = os.path.basename(bvec.path).replace('.bvec', '.nii.gz')
        if dwi_filename not in associations.keys():
            associations[dwi_filename] = {"bvec": bvec.path}
        else:
            associations[dwi_filename]["bvec"] = bvec.path

    # Associate field maps
    for fmap in fmaps:
        metadata = get_metadata(fmap)
        if isinstance(metadata.get('IntendedFor', ''), list):
            intended = metadata.get('IntendedFor', '')
        else:
            intended = [metadata.get('IntendedFor', '')]

        for target in intended:
            dwi_filename = os.path.basename(target)
            if dwi_filename not in associations.keys():
                associations[dwi_filename] = {'fmap': [fmap]}
            elif 'fmap' in associations[dwi_filename].keys():
                associations[dwi_filename]['fmap'].append(fmap)
            else:
                associations[dwi_filename]['fmap'] = [fmap]

    # Associate sbref
    for sbref in sbrefs:
        dwi_filename = os.path.basename(sbref.path).replace('sbref', 'dwi')
        if dwi_filename not in associations.keys():
            associations[dwi_filename] = {'sbref': [sbref]}
        elif 'sbref' in associations[dwi_filename].keys():
            associations[dwi_filename]['sbref'].append(sbref)
        else:
            associations[dwi_filename]['sbref'] = [sbref]

    return associations


def get_data(nSub, dwi, t1s, associations, default_readout, clean):
    """ Return subject data

    Parameters
    ----------
    nSub : String
        Subject name

    dwi : list of BIDSFile object
        DWI objects

    t1s : List of BIDSFile object
        List of T1s associated to the current subject

    associations : Dictionnary
        Dictionnary containing files associated to the DWI

    default_readout : Float
        Default readout time

    Returns
    -------
    Dictionnary containing the metadata
    """
    bvec_path = ['todo'] * len(dwi)
    bval_path = ['todo'] * len(dwi)
    dwi_path = ['todo'] * len(dwi)
    PE = ['todo'] * len(dwi)
    topup = [''] * len(dwi)

    nSess = 0
    if 'session' in dwi[0].get_entities().keys():
        nSess = dwi[0].get_entities()['session']

    nRun = 0
    if 'run' in dwi[0].get_entities().keys():
        nRun = dwi[0].get_entities()['run']

    for index, curr_dwi in enumerate(dwi):
        dwi_path[index] = dwi.path

        fmaps = []
        sbref = []
        if curr_dwi.filename in associations.keys():
            if "bval" in associations[curr_dwi.filename].keys():
                bval_path[index] = associations[curr_dwi.filename]['bval']
            if "bvec" in associations[curr_dwi.filename].keys():
                bvec_path[index] = associations[curr_dwi.filename]['bvec']
            if "fmap" in associations[curr_dwi.filename].keys():
                fmaps[index] = associations[curr_dwi.filename]['fmap'])
            if "sbref" in associations[curr_dwi.filename].keys():
                sbref[index] = associations[curr_dwi.filename]['sbref'])

            dwi_PE = 'todo'
            dwi_revPE = -1
            conversion = {"i": "x", "j": "y", "k": "z"}
            dwi_metadata = get_metadata(curr_dwi)
            if 'PhaseEncodingDirection' in dwi_metadata:
                dwi_PE = dwi_metadata['PhaseEncodingDirection']
                dwi_PE = dwi_PE.replace(dwi_PE[0], conversion[dwi_PE[0]])
                if len(dwi_PE) == 1:
                    PE[index] = dwi_PE + '-'
                else:
                    PE[index] = dwi_PE[0]
            elif clean or len(dwi) > 1:
                return {}

        # Find b0 for topup, take the first one
        # Check fMAP
        if fmaps
        rev_topup = ''
        totalreadout = default_readout
        if len(fmaps) == 0:
            if 'TotalReadoutTime' in dwi_metadata:
                totalreadout = dwi_metadata['TotalReadoutTime']
        else:
            for nfmap in fmaps:
                nfmap_metadata = get_metadata(nfmap)
                if 'PhaseEncodingDirection' in nfmap_metadata:
                    fmap_PE = nfmap_metadata['PhaseEncodingDirection']
                    fmap_PE = fmap_PE.replace(fmap_PE[0], conversion[fmap_PE[0]])
                    if fmap_PE == dwi_revPE:
                        if 'TotalReadoutTime' in dwi_metadata:
                            if 'TotalReadoutTime' in nfmap_metadata:
                                dwi_RT = dwi_metadata['TotalReadoutTime']
                                fmap_RT = nfmap_metadata['TotalReadoutTime']
                                if dwi_RT != fmap_RT and totalreadout == '':
                                    totalreadout = 'error_readout'
                                    rev_topup = 'error_readout'
                                elif dwi_RT == fmap_RT:
                                    rev_topup = nfmap.path
                                    totalreadout = dwi_RT
                                    break
                        else:
                            rev_topup[] = nfmap.path
                            totalreadout = default_readout


    t1_path = 'todo'
    t1_nSess = []
    if not t1s and clean:
        return {}

    for t1 in t1s:
        if 'session' in t1.get_entities().keys() and\
                t1.get_entities()['session'] == nSess:
            t1_nSess.append(t1)
        elif 'session' not in t1.get_entities().keys():
            t1_nSess.append(t1)

    # Take the right T1, if multiple T1s the field must be completed ('todo')
    if len(t1_nSess) == 1:
        t1_path = t1_nSess[0].path
    elif 'run' in dwi.path:
        for t1 in t1_nSess:
            if 'run-' + str(nRun) in t1.path:
                t1_path = t1.path

    return {'subject': nSub,
            'session': nSess,
            'run': nRun,
            't1': t1_path,
            'dwi': dwi_path[0],
            'bvec': bvec_path[0],
            'bval': bval_path[0],
            'rev_dwi': dwi_path[1],
            'rev_bvec': bvec_path[1],
            'rev_bval': bval_path[1],
            'topup': topup[0],
            'rev_topup': topup[1],
            'DWIPhaseEncodingDir': PE[],
            'rev_DWIPhaseEncodingDir': PE[1],
            'TotalReadoutTime': totalreadout}


def associate_dwi(layout, nSub):
    """ Return subject data
    Parameters
    ----------
    layout:

    nSub:

    Returns
    -------
    all_dwis: list


    """
    all_dwis = []

    for curr_sess in layout.get_sessions():
        dwis = layout.get(subject=nSub,
                          session=curr_sess,
                          datatype='dwi', extension='nii.gz',
                          suffix='dwi')
        if len(dwis) == 1:
            all_dwis.append(dwi)
        elif len(dwis) > 1:
            all_runs = [curr.entities['run'] for curr_dwi in dwis if curr_dwi.entities.has_key('run') ]
            if all_runs:
                for curr_run in all_runs:
                    dwis = layout.get(subject=nSub,
                                      session=curr_sess,
                                      run=curr_run,
                                      datatype='dwi', extension='nii.gz',
                                      suffix='dwi')
                    if dwis:
                        all_dwis.append(dwis)
            else:
                all_dwis.append(dwis)

    return all_dwis


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_outputs_exist(parser, args, args.out_json)

    data = []
    layout = BIDSLayout(args.in_bids, index_metadata=False)
    subjects = layout.get_subjects()

    if args.participants_label:
        subjects = [nSub for nSub in args.participants_label if nSub in subjects]

    for nSub in subjects:
        dwis = get_dwis(layout, nSub)
        t1s = layout.get(subject=nSub,
                         datatype='anat', extension='nii.gz',
                         suffix='T1w')
        fmaps = layout.get(subject=nSub,
                           datatype='fmap', extension='nii.gz',
                           suffix='epi')

        fmaps exclude sbref HERE if sbref in name 

        bvals = layout.get(subject=nSub,
                           datatype='dwi', extension='bval',
                           suffix='dwi')
        bvecs = layout.get(subject=nSub,
                           datatype='dwi', extension='bvec',
                           suffix='dwi')
        sbrefs = layout.get(subject=nSub,
                           datatype='dwi', extension='nii.gz',
                           suffix='sbref')

        # Get associations relatives to DWIs
        associations = get_dwi_associations(fmaps, bvals, bvecs, sbrefs)

        # Get the data for each run of DWIs
        for dwi in dwis:
            data.append(get_data(nSub, dwi, t1s, associations,
                                 args.readout, args.clean))

    if args.clean:
        data = [d for d in data if d]

    with open(args.out_json, 'w') as outfile:
        json.dump(data,
                  outfile,
                  indent=4,
                  separators=(',', ': '),
                  sort_keys=True)
        # Add trailing newline for POSIX compatibility
        outfile.write('\n')


if __name__ == '__main__':
    main()

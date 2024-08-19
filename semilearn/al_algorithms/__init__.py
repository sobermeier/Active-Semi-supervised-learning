# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from semilearn.core.utils import AL_ALGORITHMS
name2alg = AL_ALGORITHMS


def get_al_algorithm(args):
    print(AL_ALGORITHMS.keys())
    print("**************")
    if args.al_algorithm in AL_ALGORITHMS:
        al_alg = AL_ALGORITHMS[args.al_algorithm]( # name2alg[args.algorithm](
            args=args,
            gpu=args.gpu,
        )
        return al_alg
    else:
        raise KeyError(f'Unknown algorithm: {str(args.al_algorithm)}')




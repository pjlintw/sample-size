# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



def main():

    emsizes = [500,100,150,200,50]
    nhids = [500,1024,500,100,100]
    nlayerss = [3,4,5,3,5,3]
    nheads = [3,3,5,4,8]

    with open('results/nas.log','w') as nas_log:
        for emsize, nhid, nlayers, nhead in zip(emsizes,nhids,nlayerss,nheads):
            model_arch = {
                'emsize': 50,
                'nhid': 100,
                'nlayers': 2,
                'nhead': 2,
            }
            best_perf,first_zen_score = search(model_arch)
            nas_log.write(f'best_perf: {best_perf}, zen_score: {first_zen_score}, emsize: {emsize}, nhid: {nhid}, nlayers: {nlayers}, nhead: {nhead}\n')



if __name__ == "__main__":
    main()

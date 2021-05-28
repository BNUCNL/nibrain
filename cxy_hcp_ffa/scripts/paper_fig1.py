from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir,
                 'analysis/s2/1080_fROI/refined_with_Kevin/paper_fig/fig1')


def get_concat_h(im1, im2):
    # https://note.nkmk.me/en/python-pillow-concat-images/
    from PIL import Image

    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def individual_exemplars(hemi='rh'):
    from PIL import Image

    hemi2box = {
        'lh': (162, 0, 314, 370),
        'rh': (0, 0, 162, 370)
    }
    activ_fig_files = pjoin(proj_dir,
                            'analysis/s2/1080_fROI/fig/{0}_1_{1}.jpg')
    roi_fig_files = pjoin(proj_dir, 'analysis/s2/1080_fROI/'
                          'refined_with_Kevin/rois_v3_fig/{0}_2_{1}.jpg')
    out_files = pjoin(work_dir, '{0}_{1}_{2}.jpg')

    box = hemi2box[hemi]
    subj_indices = (1, 2, 3, 21, 68, 70, 80, 316, 535)
    subj_ids = ('100307', '100408', '100610', '103818', '114116',
                '114318', '116221', '163129', '211417')

    for idx, subj_id in zip(subj_indices, subj_ids):
        activ_img = Image.open(activ_fig_files.format(idx, subj_id))
        roi_img = Image.open(roi_fig_files.format(idx, subj_id))

        img = get_concat_h(activ_img.crop(box), roi_img.crop(box))
        img.save(out_files.format(idx, hemi, subj_id))


if __name__ == '__main__':
    individual_exemplars(hemi='rh')

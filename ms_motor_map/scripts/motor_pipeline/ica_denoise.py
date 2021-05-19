import os, subprocess, argparse
from os.path import join as pjoin
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from email.mime.text import MIMEText
from email.header import Header
from smtplib import SMTP_SSL


def send_mail(receiver, mail_title, mail_content):
    host_server = 'smtp.qq.com'
    sender_qq = '1045418215@qq.com'
    pwd = 'dflxxfxcrwkybfeg'
    # ssl
    smtp = SMTP_SSL(host_server)
    # set_debuglevel()
    smtp.set_debuglevel(1)
    smtp.ehlo(host_server)
    smtp.login(sender_qq, pwd)
    msg = MIMEText(mail_content, "plain", 'utf-8')
    msg["Subject"] = Header(mail_title, 'utf-8')
    msg["From"] = sender_qq
    msg["To"] = receiver
    smtp.sendmail(sender_qq, receiver, msg.as_string())
    smtp.quit()


def run_ica_denoise(args):
    # collect waiting runs
    fmriprep_dir = pjoin(args.projectdir, 'data/bold/derivatives/fmriprep')
    melodic_dir = pjoin(args.projectdir, 'data/bold/derivatives/melodic')
    sub_dir, ses_dir = args.input.split('_')[0], args.input.split('_')[1]
    ica_output = pjoin(melodic_dir, "%s/%s/%s.ica" % (sub_dir, ses_dir, args.input))
    try:
        # make original files dir
        mix_orignal_dir = pjoin(ica_output, 'series_original')
        nii_orignal_dir = pjoin(ica_output, 'spatial_original')
        if not os.path.exists(mix_orignal_dir):
            os.makedirs(mix_orignal_dir)
            print('[Node] os.makedirs({})'.format(mix_orignal_dir))
            if not os.path.exists(pjoin(mix_orignal_dir, 'melodic_mix')):
                # copy original file
                _ = ' '.join(['mv', pjoin(ica_output, 'melodic_mix'), mix_orignal_dir])
                subprocess.check_call(_, shell=True)
                print('[Node]', _)
            if not os.path.exists(pjoin(mix_orignal_dir, 'melodic_FTmix')):
                # copy original file
                _ = ' '.join(['mv', pjoin(ica_output, 'melodic_FTmix'), mix_orignal_dir])
                subprocess.check_call(_, shell=True)
                print('[Node]', _)
        if not os.path.exists(nii_orignal_dir) and args.fwhm:
            os.makedirs(nii_orignal_dir)
            print('[Node] os.makedirs({})'.format(nii_orignal_dir))
            if not os.path.exists(pjoin(nii_orignal_dir, 'melodic_IC.nii.gz')):
                # copy original file
                _ = ' '.join(['mv', pjoin(ica_output, 'melodic_IC.nii.gz'), nii_orignal_dir])
                subprocess.check_call(_, shell=True)
                print('[Node]', _)

        # load & create
        df_mix = pd.read_csv(pjoin(mix_orignal_dir, 'melodic_mix'), sep='  ', header=None)
        df_ft = pd.read_csv(pjoin(mix_orignal_dir, 'melodic_FTmix'), sep='  ', header=None)

        # name the columns
        df_mix.columns = [_ for _ in range(df_mix.shape[-1])]
        df_ft.columns = [_ for _ in range(df_ft.shape[-1])]

        # output ftmix, rows will change for frequency power
        df_FT = pd.DataFrame(data=np.zeros((int(df_mix.shape[0] / 2) - 1,
                    df_mix.shape[-1])), columns=[_ for _ in range(df_mix.shape[-1])])

        # prepare regressors matrix
        # polyfit
        x = np.linspace(1, df_mix.shape[0], df_mix.shape[0])
        X = np.vstack(tuple([x ** _ for _ in range(args.order + 1)]))
        # motion
        if args.motion:
            confoundcsv = pjoin(fmriprep_dir, "%s/%s/func/%s_desc-confounds_timeseries.tsv" \
                                % (sub_dir, ses_dir, args.input))
            # motion
            df_conf = pd.read_csv(confoundcsv, sep='\t')
            motion = np.vstack(tuple([np.array(df_conf[_]).astype(np.float64) for _ in \
                                      ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']]))
            X = np.vstack((X, motion))
        X = X.transpose()

        for __ in range(df_mix.shape[-1]):
            # dependent variable
            y = np.array(df_mix[__])

            # polyfit with motion
            reg = LinearRegression().fit(X, y)
            res = y - reg.predict(X)

            # frequency spetrum
            signal = np.array(res, dtype=float)
            power = np.abs(np.fft.fft(signal)) ** 2
            n = int(signal.size)
            timestep = 2
            freqs = np.fft.fftfreq(n, timestep)
            power = power[freqs > 0]
            freqs = freqs[freqs > 0]
            idx = np.argsort(freqs)

            # write into DataFrame
            df_mix[__] = np.float32(res)
            df_FT[__] = np.float32(power[idx])

            print('IC {} done'.format(__))

        # generate
        if args.overwrite:
            df_mix.to_csv(pjoin(ica_output, 'melodic_mix'), sep=' ', columns=None,
                          header=None, index=False)
            df_FT.to_csv(pjoin(ica_output, 'melodic_FTmix'), sep=' ', columns=None,
                         header=None, index=False)
        else:
            if os.path.exists(pjoin(ica_output, 'melodic_mix')):
                print('already exist: melodic_mix & melodic_FTmix')
                raise AssertionError('already exist: melodic_mix & melodic_FTmix')
            else:
                df_mix.to_csv(pjoin(ica_output, 'melodic_mix'), sep=' ', columns=None,
                            header=None, index=False)
                df_FT.to_csv(pjoin(ica_output, 'melodic_FTmix'), sep=' ', columns=None,
                            header=None, index=False)

        # spatial smooth
        if args.fwhm:
            melodic_IC = pjoin(nii_orignal_dir, 'melodic_IC.nii.gz')
            fslmaths_command = ' '.join(
                ['fslmaths', melodic_IC, '-s', str(args.fwhm / 2.355), pjoin(ica_output, 'melodic_IC.nii.gz')])
            subprocess.check_call(fslmaths_command, shell=True)
            print(fslmaths_command)

        if args.email_address:
            send_mail(args.email_address, 'congratulations',
                      'ICA denoise {} sucessfully done'.format(args.input))
    except Exception:
        print(Exception)
        if args.email_address:
            send_mail(args.email_address, 'sorry',
                      'ICA denoise {} has some problems as follows:\n {}'.format(args.input, str(Exception)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="bids name pattern of the ica run")
    parser.add_argument("projectdir", help="path of project dir")

    parser.add_argument("--overwrite", action="store_true",
                        help="if chosen, the exists files will be overwrite without assertion")
    parser.add_argument('-d', "--order", type=int,
                        help="the order of polyfit, practically it should be less than half of the run duration.")
    parser.add_argument('-m', "--motion", action="store_true",
                        help="if chosen, motion regressor will be added in to denoise")
    parser.add_argument('-s', "--fwhm", type=float, help="the full width at the half maximum")
    parser.add_argument('-e', "--email-address",
                        help="if given, a message will be sent to the address once the procedure succeeds or goes wrong")
    args = parser.parse_args()

    run_ica_denoise(args)

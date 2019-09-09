# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram

def make_figfunction(figuretype, isshow = True):
    """
    A function to pack figure factory, make it easier to use
 
    Parameters:
    -----------
    figuretype: 'corr', correlation plots
                'mat', matrix plot
                'bar', plot bar
                'hist', histogram
                'hierarchy', hierarchy maps
                'line', line maps
                'scatter', scatter maps
                'violin', violin maps
                'radar', radar chart
                'montage', montage maps
    isshow: whether to show plot or not. 
            if isshow is False, output plot can be combined together
    Return:
    -------
    figinstance: figure function

    Example:
    --------
    >>> plotcorr = make_figfunction('corr')
    """
    figFact = _FigureFactory(isshow = isshow)
    return figFact.createfactory(figuretype)

class _FigureFactory(object):
    """
    A Factory for Figures
    ----------------------------
    Example:
        >>> figFact = plotfig.FigureFactory()
        >>> plotmat = figFact.createfactory('mat')
    """
    def __init__(self, isshow = True):
        self._isshow = isshow

    def __str__(self):
        return 'A factory for plotting figures'

    def createfactory(self, figuretype):
        """
        Create factory by this function
        ------------------------------------
        Parameters:
            figuretype: 'corr', correlation plots
                        'mat', matrix plot
                        'bar', plot bar
                        'hist', histogram
                        'hierarchy', hierarchy maps
                        'line', line maps
                        'scatter', scatter maps
                        'violin', violin maps
                        'radar', radar chart
                        'montage', montage maps
        """
        figure = self._Figures(isshow = self._isshow)
        if figuretype == 'corr':
            figuror = figure._corr_plotting
        elif figuretype == 'mat':
            figuror = figure._mat_plotting
        elif figuretype == 'bar':
            figuror = figure._bar_plotting
        elif figuretype == 'hist':
            figuror = figure._hist_plotting
        elif figuretype == 'hierarchy':
            figuror = figure._hierarchy_plotting
        elif figuretype == 'line':
            figuror = figure._simpleline_plotting
        elif figuretype == 'scatter':
            figuror = figure._scatter_plotting
        elif figuretype == 'violin':
            figuror = figure._violin_plotting
        elif figuretype == 'radar':
            figuror = figure._radar_plotting
        elif figuretype == 'montage':
            figuror = figure._montage_plotting
        else:
              raise Exception('wrong parameter input!')
        return figuror

    class _Figures(object):
        def __init__(self, labelsize = 14, isshow = True):
            plt.rc('xtick', labelsize = labelsize)
            plt.rc('ytick', labelsize = labelsize)
            self._labelsize = labelsize
            self._isshow = isshow

        def _corr_plotting(self, meas1, meas2, labels=['',''], method = 'pearson'):
            """
            Make scatter plot and give a fit on it.
            ------------------------------------------
            Paramters:
                meas1: feature measurement
                meas2: feature measurement
                labels: A list contains two labels.
                        labels[0] means label of meas1, labels[1] means label of meas2.
                method: 'pearson' or 'spearman' correlation
            Example:
                >>> plotcorr(data1, data2, labels = label, method = 'pearson')
            """
            meas1 = np.array(meas1)
            meas2 = np.array(meas2)
            if (meas1.dtype != 'O') | (meas2.dtype != 'O'):
                samp_sel = ~np.isnan(meas1*meas2)
                x = meas1[samp_sel]
                y = meas2[samp_sel]
            else:
                x = meas1
                y = meas2
            if method == 'pearson':
                corr, pval = stats.pearsonr(x, y)
            elif method == 'spearman':
                corr, pval = stats.spearmanr(x, y)
            else:
                raise Exception('Wrong method you used')
            fig, ax = plt.subplots()
            plt.scatter(x, y)
            plt.plot(x, np.poly1d(np.polyfit(x,y,1))(x))
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            ax.set_aspect((x1-x0)/(y1-y0))
            ax.text(x0+0.1*(x1-x0), y0+0.9*(y1-y0), 'r = %.3f, p = %.3f' % (corr, pval))
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            plt.title(method.capitalize()+' Correlation')

            if self._isshow is True:
                plt.show()

        def _mat_plotting(self, data, xlabel='', ylabel=''):
            """
            Plot matrix using heatmap
            ------------------------------------
            Paramters:
                data: raw data
                xlabel: xlabels
                ylabel: ylabels
            Example:
                >>> plotmat(data, xlabel = xlabellist, ylabel = ylabellist)
            """
            sns.heatmap(data, xticklabels = xlabel, yticklabels = ylabel)
            if self._isshow is True:
                plt.show()

        def _bar_plotting(self, data, title = '', xlabels = '', ylabels = '', legendname = None, legendpos = 'upper left', rotation = 'vertical', err=None):
            """
            Do barplot
            --------------------------
            Parameters:
                data: raw data
                title: title of figures
                xlabels, ylabels: xlabel and ylabel of figures
                legendname: identified legend name
                legendpos: by default is 'upper left'
                rotation: by default is 'vertical', you can write degree number to control label angles.
                err: error of data estimation. Used for errorbar
            Example:
                >>> plotbar(data, title = titletxt, xlabels = xlabel, ylabels = ylabel, legendname = legendnametxt, err = errdata)
            """
            color = ['#BDBDBD', '#575757', '#404040', '#080808', '#919191']
            if isinstance(data, list):
                data = np.array(data)
            if data.ndim == 1:
                data = np.expand_dims(data, axis = 1)
            ind = np.arange(data.shape[0])
            width = 0.70/data.shape[1]
            fig, ax = plt.subplots() 
            rects = []
            if err is None:
                for i in range(data.shape[1]):
                    if legendname is None:
                        rects = ax.bar(ind + i*width, data[:,i], width, color = color[i%5])
                    else:
                        rects = ax.bar(ind + i*width, data[:,i], width, color = color[i%5], label = legendname[i])
                    ax.legend(loc=legendpos) 
            else:
                if isinstance(err, list):
                    err = np.array(err)
                if err.ndim == 1:
                    err = np.expand_dims(err, axis = 1)
                for i in range(data.shape[1]):
                    try:
                        rects = ax.bar(ind + i*width, data[:,i], width, color = color[i%5], yerr = err[:,i], error_kw=dict(ecolor = '#757575', capthick=1), label = legendname[i])
                    except TypeError:
                        rects = ax.bar(ind + i*width, data[:,i], width, color = color[i%5], yerr = err[:,i], error_kw=dict(ecolor = '#757575', capthick=1), label = legendname)                     
                    ax.legend(loc=legendpos)
            
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            ax.set_aspect((x1-x0)/(y1-y0))
            ax.set_ylabel(ylabels)
            ax.set_xticks(ind + width*i/2)
            if (np.min(data)<0) & (np.max(data)<0):
                ax.set_ylim([1.33*np.min(data), 0])
            elif (np.min(data)<0) & (np.max(data)>0):
                ax.set_ylim([1.33*np.min(data), 1.33*np.max(data)])
            else:
                ax.set_ylim([0, 1.33*np.max(data)])
            plt.xticks(rotation=rotation)
            ax.set_xticklabels(xlabels)
            ax.set_title(title, fontsize=12)

            if self._isshow is True:
                plt.show()

        def _hist_plotting(self, n_scores, legend_label, *oppar):
            """
            Plot histogram of given data
            Parameters:
            ----------------------------------
                n_scores: scores
                legend_label: data legend label
                score: Optional choice. used for permutation cross validation results.In permutation cross validation, n_scores means value of permutation scores, score means actual score.
                pval: Optional choice. Need to use with score. p values of permutation test.
            Example:
                >>> plothist(values, legend_label = labels, score = score_collect, pval = pvalue)
            """
            if isinstance(legend_label, str):
                legend_label = [legend_label]
            if len(oppar) == 0:
                plt.hist(n_scores, 50, label = legend_label, color='gray')
                ylim = plt.ylim()
            elif len(oppar) == 2:
                plt.hist(n_scores, 50, label = legend_label, color='gray')
                ylim = plt.ylim()
                if oppar[1]<0.001:
                    plt.plot(2*[oppar[0]], ylim, '--k', linewidth = 3,
                             label = 'Pvalue<0.001')
                else:
                    plt.plot(2*[oppar[0]], ylim, '--k', linewidth = 3,
                             label = 'Pvalue %.3f' % oppar[1])
                plt.ylim(ylim)
            else:
                raise Exception('parameter numbers should be 2 or 4!')
            plt.legend()
            plt.xlabel('Score')
             
            if self._isshow is True:
                plt.show()
          
        def _hierarchy_plotting(self, distance, regions):
            """
            Plot hierarchy structure of specific indices between regions
            -------------------------------
            Parameters:
                distance: distance array, distance array by using scipy.pdist
                regions: region name       
            Example:
                >>> plothierarchy(distance, regions) 
            """
            Z = linkage(distance, 'average')
            dendrogram(Z, labels = regions)

            if self._isshow is True:
                plt.show()

        def _simpleline_plotting(self, x, y, yerr = None, xlabel='', ylabel='', legend = None, xlim = None, ylim = None, scaling = False):
            """
            Plot an array using simple lines
            For better showing, rescaling each array into range of 0 to 1
            --------------------------------------
            Parameters:
                x: range of x-axis
                y: data array, a x*y array, y means number of lines
                xlabel: xlabel
                ylabel: ylabel
                legend: legend
                xlim: By default is None, if ylim exists, limit x values of figure, [xmin, xmax]
                ylim: By default is None, if ylim exists, limit y values of figure, [ymin, ymax]
                scaling: whether do rescaling or not to show multiple lines
            Example:
                >>> plotline(x, y, yerr)
            """
            if y.ndim == 1:
                y = np.expand_dims(y, axis = -1)
            if yerr is not None:
                if yerr.ndim == 1:
                    yerr = np.expand_dims(yerr, axis = -1)
            fig, ax = plt.subplots()
            COLORNUM = y.shape[-1]
            cm = plt.get_cmap('Dark2')
            ax.set_color_cycle([cm(1.0*i/COLORNUM) for i in range(COLORNUM)])
            if scaling is True:
                y_scaling = np.zeros_like(y)
                for i in range(y.shape[1]):
                    y_scaling[:,i] = (y[:,i]-np.min(y[:,i]))/(np.max(y[:,i])-np.min(y[:,i]))
            else:
                y_scaling = y
            for i in range(y_scaling.shape[-1]):
                if yerr is not None:
                    plt.errorbar(x, y_scaling[:,i], yerr = yerr[:,i])
                else:
                    plt.errorbar(x, y_scaling[:,i], yerr = yerr)
            if legend is not None:
                plt.legend(legend, fontsize = 14)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)
            if self._isshow is True:
                plt.show()

        def _scatter_plotting(self, array1, array2, xlabel='', ylabel='', colors = ['red'], xlim = None, ylim = None):
            """
            Plot scatter map among several group's data
            ----------------------------------------------
            Parameters:
                array1: axis x data. m*n arrays, n means different groups
                array2: axis y data. array2 should have same shape with array1
                xlabel: xlabel
                ylabel: ylabel
                colors: color of each group
                xlim: axis x limitation, [xmin, xmax]
                ylim: axis y limitation, [ymin, ymax]
            Example:
                >>> plotscatter(array1, array2)
            """
            if isinstance(array1, list):
                array1 = np.array(array1)
            if isinstance(array2, list):
                array2 = np.array(array2)
            if array1.ndim == 1:
                array1 = np.expand_dims(array1, axis=1)
            if array2.ndim == 1:
                array2 = np.expand_dims(array2, axis=1)
            assert array1.shape == array2.shape, 'arrays shape should be equal'
            assert array1.shape[1] == len(colors), 'data class need to be equal with color class'
            for i,c in enumerate(colors):
                plt.scatter(array1[:,i], array2[:,i], color = c)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)

            if self._isshow is True:
                plt.show()

        def _violin_plotting(self, data, xticklabels = None, showmeans = True, xlabel = '', ylabel = '', ylim = None):
            """
            Plot violin figures by a 2D data
            ----------------------------------
            Parameters:
            data: a 2 dimensional data, M*N, where N is the number of category
            xticklabels: xticklabels, by default is None
            showmeans: whether to show means in violin plot
            xlabel: xlabel
            ylabel: ylabel
            ylim: limitation of y

            Examples:
            ----------
            >>> plotviolin(data)
            """
            assert data.ndim == 2, 'A two-dimension data should be inputted'
            cat_num = data.shape[-1]
            ax = plt.subplot()                       
            plt.violinplot(data, np.arange(1, cat_num+1), showmeans = showmeans)
            ax.set_xticks(np.arange(1, cat_num+1))
            if xticklabels is not None:
                ax.set_xticklabels(xticklabels, fontsize = self._labelsize)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if ylim is not None:
                plt.ylim(ylim)
            
            if self._isshow is True:
                plt.show()

        def _radar_plotting(self, data, xlabel = '', ylabel = '', grp_label = '', fontsize = 12):
            """
            Show radar chart
        
            Parameters:
            ------------
            data: M*N series point, not recommend to read more than 3 groups in one radar chart. N means the number of group.
            xlabel: label in x-axis
            ylabel: label in y-axis
            grp_label: group label
            fontsize: fontsize
            """
            if isinstance(data, list) is True:
                data = np.array(data)
            if data.ndim == 1:
                data = np.expand_dims(data, axis=-1)
            pt_N = data.shape[0]
            grp_N = data.shape[1]
            if grp_label == '':
                grp_label = ['']*grp_N
            # Background
            angles = [n/float(pt_N)*2*np.pi for n in range(pt_N)]
            angles += angles[:1]

            ax = plt.subplot(111, polar=True)
            # Set theta direction and its offset
            ax.set_theta_offset(np.pi/2)
            ax.set_theta_direction(-1)
            # Draw xlabels
            plt.xticks(angles[:-1], xlabel, color = 'black', fontsize=fontsize)
            # Draw ylabels
            ax.set_rlabel_position(0)
            ax.set_ylabel(ylabel, color='black', fontsize=fontsize)
            if (np.min(data)<0) & (np.max(data)<0):
                ax.set_ylim([1.33*np.min(data), 0])
            elif (np.min(data)<0) & (np.max(data)>0):
                ax.set_ylim([1.33*np.min(data), 1.33*np.max(data)])
            else:
                ax.set_ylim([0, 1.33*np.max(data)])

            # Add plots
            for i in range(grp_N):
                tmp_data = data[:,i].tolist()
                tmp_data += tmp_data[:1]
                ax.plot(angles, tmp_data, linewidth=1, linestyle='solid', label=grp_label[i])
                ax.fill(angles, tmp_data, 'b', alpha=0.1)

            # Add legend
            if grp_N > 1:
                ax.legend(grp_label, loc=[0.9,0.9])
            
            if self._isshow is True: 
                plt.show()
          

        def _montage_plotting(self, pic_path, column_num, row_num, text_list = None, text_loc = (0,50), fontsize = 12, fontcolor = 'w'):
            """
            Show pictures in a figure

            Parameters:
            -----------
            pic_path: path of pictures, as a list
            column_num: picture numbers shown in each column
            row_num: picture numbers shown in each row
            text_list: whether to show text in each picture, by default is None
            text_loc: text location
            fontsize: text font size
            fontcolor: text font color

            Example:
            ---------
            >>> plotmontage(pic_path, 8, 6, text_list)
            """
            try:
                from skimage import io as sio
            except ImportError as e:
                raise Exception('Please install skimage first')

            assert (len(pic_path) < column_num*row_num)|(len(pic_path) == column_num*row_num), "Number of pictures is larger than what subplot could accomdate, please increase column_num/row_num values"
            if text_list is not None:
                assert len(pic_path) == len(text_list), "pic_path shall have the same length as text_list"
                 
            fig = plt.figure(figsize=(2*column_num, 2*row_num))
            for i, ppath in enumerate(pic_path):
                img = sio.imread(ppath)
                ax = plt.subplot(row_num, column_num, i+1)
                ax.set_axis_off()
                plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.1, hspace=0.1)
                if text_list is not None:
                    plt.text(text_loc[0], text_loc[1], text_list[i], fontsize=fontsize, color = fontcolor)
                plt.imshow(img)

            if self._isshow is True:
                plt.show()









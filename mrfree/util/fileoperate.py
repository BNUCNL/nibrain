# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import os
import shutil

verbose = True

class DeleteFile(object):
    """
    Delete file/directory
    --------------------------------
    Parameters:
        path: path for deleting
        confirm: by default is 'no'.
                 if 'no', each time when deleting file will be asked
                 if 'yes', delete file without confirming
    >>> m = DeleteFile()
    >>> m.exexute(path)
    """
    def __init__(self):
        pass
    def execute(self, path, confirm = 'no'):
        if confirm == 'no':
            confirm = input('Are you sure to delete {}?'.format(path))
        if confirm == 'yes':
            if verbose:
                print('Deleting {}'.format(path))
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
        elif confirm == 'no':
            print('Not delete {}'.format(path))
        else:
            raise Exception('Wrong confirm parameter')
    def undo(self):
        pass

class CopyFile(object):
    """
    Copy file/directory from source to destination
    ------------------------------------------------
    Parameters:
        path_src: source path
        path_dst: destination path

    >>> m = CopyFile()
    >>> m.execute(src, dst)
    """
    def __init__(self):
        pass
    def execute(self, src, dst, dst_filename = None):
        if verbose:
            print('Copying {0} to {1}'.format(src, dst))
        assert os.path.isdir(dst), "determine file name in parameter of dst_filename."
        if os.path.isfile(src):
            if dst_filename is None:
                shutil.copyfile(src, os.path.join(dst, [x for x in src.split('/') if x][-1]))
            else:
                shutil.copyfile(src, os.path.join(dst, dst_filename))
        else:
            shutil.copytree(src, os.path.join(dst, [x for x in src.split('/') if x][-1]))
    def undo(self):
        pass
    
class RenameFile(object):
    """
    Rename file or directory    
    ---------------------------
    Parameters:
        src: source file
        dst: destination file

    >>> m = RenameFile()
    >>> m.execute(src, dst)
    """
    def __init__(self):
        pass
    def execute(self, src, dst):
        if verbose:
            print('Renaming {0} to {1}'.format(src, dst))
        os.rename(src, dst)
    def undo(self, src, dst):
        if verbose:
            print('renaming {0} to {1}'.format(dst, src))
        os.rename(dst, src)
  
class CreateFile(object):
    """
    Create file/directory
    ---------------------------
    Parameters:
        path: data path
              if it's a directory, create directory, 
              else create a file
        text: by default a text

    >>> a = CreateFile()
    >>> a.execute(path, text)
    """
    def __init__(self):
        pass
    def execute(self, path, text = 'Hello, world\n'):
        if verbose:
            print('Creating {}'.format(path))
        if os.path.isfile(path):
            with open(path, 'w') as out_file:
                out_file.write(text)
        else:
            os.mkdir(path)
    def undo(self):
        pass


         








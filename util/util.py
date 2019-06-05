"""
Random utility functions used by other parts of mlopt.
"""
import os
import pathlib
import shutil
import tarfile
import urllib
import zipfile


class cachedproperty(object):
    """The ``cachedproperty`` is used similar to :class:`property`, except
    that the wrapped method is only called once. This is commonly used
    to implement lazy attributes.

    After the property has been accessed, the value is stored on the
    instance itself, using the same name as the cachedproperty. This
    allows the cache to be cleared with :func:`delattr`, or through
    manipulating the object's ``__dict__``.
    """
    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.__isabstractmethod__ = getattr(func, '__isabstractmethod__', False)
        self.func = func

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value

    def __repr__(self):
        cn = self.__class__.__name__
        return '<%s func=%s>' % (cn, self.func)


def download_data(out_path, url, extract=True, force=False):
    pathlib.Path(out_path).mkdir(exist_ok=True)
    out_filename = os.path.join(out_path, os.path.basename(url))

    if os.path.isfile(out_filename) and not force:
        print(f'File {out_filename} exists, skipping download.')
    else:
        print(f'Downloading {url}...')

        with urllib.request.urlopen(url) as response:
            with open(out_filename, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

        print(f'Saved to {out_filename}.')

    extracted_dir = None
    if extract and out_filename.endswith('.zip'):
        print(f'Extracting {out_filename}...')
        with zipfile.ZipFile(out_filename, "r") as zipf:
            names = zipf.namelist()
            zipf.extractall(out_path)
            zipinfos = zipf.infolist()
            first_dir = next(filter(lambda zi: zi.is_dir(), zipinfos)).filename
            extracted_dir = os.path.join(out_path, os.path.dirname(first_dir))
            print(f'Extracted {len(names)} to {extracted_dir}')
            retval = extracted_dir

    if extract and out_filename.endswith(('.tar.gz', '.tgz')):
        print(f'Extracting {out_filename}...')
        with tarfile.TarFile(out_filename, "r") as tarf:
            members = tarf.getmembers()
            tarf.extractall(out_path)
            first_dir = next(filter(lambda ti: ti.isdir(), members)).name
            extracted_dir = os.path.join(out_path, os.path.dirname(first_dir))
            print(f'Extracted {len(members)} to {extracted_dir}')
            retval = extracted_dir

    return out_filename, extracted_dir


def import_name(full_name):
    module_name, unit_name = full_name.rsplit('.', 1)
    return getattr(__import__(module_name, fromlist=['']), unit_name)

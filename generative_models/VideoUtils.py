import subprocess
import threading
from os import path,remove,devnull
from sys import stdin,stdout
import numpy as np
 
class VideoSink(object) :
        """
        VideoSink: write numpy array to a movie file
        ------------------------------------------------------------------------
 
        Requires mencoder:
        <http://www.mplayerhq.hu/design7/dload.html>
 
        Parameters:
        ----------------
        filename        string  The path/name of the output file
 
        size            tuple   The row/column dimensions of the output movie
 
        rate            scalar  The framerate (fps)
 
        colorspace      string  The color space of the output frames, 8-bit RGB
                                by default ('$ mencoder -vf format=fmt=help'
                                for a list of valid color spaces)
 
        codec           string  The codec to use, libavcodecs by default
                                ('$ mencoder -ovc -h' for a list of valid codecs)
 
        Methods:
        ----------------
        VideoSink(array)        Write the input array to the specified .avi file
                                - must be in the form [rows,columns,(channels)],
                                and (rows,colums) must match the 'size'
                                specified when initialising the VideoSink.
 
        VideoSink.close()       Close the .avi file. The file *must* be closed
                                after writing, or the header information isn't
                                written correctly
 
        Example useage:
        ----------------
        frames = np.random.random_integers(0,255,size=50,100,200,3).astype('uint8')
        vsnk = VideoSink('mymovie.avi',size=frames.shape[1:3],colorspace='rgb24')
        for frame in frames:
                vsnk(frame)
        vsnk.close()
 
        Alistair Muldal, Aug 2012
 
        Credit to VokkiCodder for this idea
        <http://vokicodder.blogspot.co.uk/2011/02/numpy-arrays-to-video.html>
 
        """
        def __init__( self, filename='output.avi', size=(512,512), rate=25, colorspace='rgb24',codec='lavc'):
 
                # row/col --> x/y by swapping order
                self.size = size[::-1]
 
                cmdstring  = (  'mencoder',
                                '/dev/stdin',
                                '-demuxer', 'rawvideo',
                                '-rawvideo', 'w=%i:h=%i'%self.size+':fps=%i:format=%s'%(rate,colorspace),
                                '-o', filename,
                                '-ovc', codec,
                                '-nosound',
                                '-really-quiet'
                                )
                self.p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE, shell=False)
 
        def __call__(self, frame) :
                assert frame.shape[0:2][::-1] == self.size
                # frame.tofile(self.p.stdin) # should be faster but it is indeed slower
                self.p.stdin.write(frame.tostring())
        def close(self) :
                self.p.stdin.close()
                self.p.terminate()
 
class VideoSource(object):
        """
        VideoSource: create numpy arrays from frames in a movie file
        ------------------------------------------------------------------------
 
        Requires mencoder:
        <http://www.mplayerhq.hu/design7/dload.html>
 
        Parameters:
        ----------------
        filename        string  The path/name of the output file.
 
        colorspace      string  The color space of the output frames, 8-bit RGB
                                by default*.
 
        cache           string  Temporary cache for the raw frames,
                                '/tmp/tmpframes_<filename>' by default. This
                                file is deleted when .close() is called.
 
        blocking        bool    If True, VideoSource will block control until
                                all of the source frames have been decoded
                                (useful if VideoSource is being created within a
                                function). By default it will return control
                                immediately, but frames will not be accessible
                                until decoding is complete.
       
        *Only the following colorspaces are currently supported:
        8-bit monochrome:       'y800','y8'
        8-bit RGB:              'rgb24','bgr24'
        8-bit RGBA:             'rgba','argb','rgb32','bgra','abgr','bgr32'
 
        Methods:
        ----------------
        VideoSource[1:10:2]     Grab frames from the source movie as numpy
                                arrays. Currently only a single set of indices
                                is supported. It is also possible to iterate
                                over frames.
 
        VideoSource.close()     Closes and deletes the temporary frame cache
 
        VideoSink.decode()      Decode the frames again and create a new cache.
                                Do this if you called .close() and want to
                                access the frames again.
 
        Example useage:
        ----------------
        vsrc = VideoSource('mymovie.avi',blocking=True)
        oddframes = vsrc[1::2]                  # get item(s)
        framelist = [frame for frame in vsrc]   # iterate over frames
        vsrc.close()                            # close and delete cache
 
        Alistair Muldal, Aug 2012
 
        Inspired by VokkiCodder's VideoSink class
        <http://vokicodder.blogspot.co.uk/2011/02/numpy-arrays-to-video.html>
 
        """
        def __init__(self,filename,colorspace='rgb24',cache=None,blocking=False):
 
                # check that the file exists
                try:
                        open(filename)
                except IOError:
                        raise IOError('Movie "%s" does not exist!' %filename)
 
                # default cache location
                if cache == None:
                        cache = '/tmp/%s_tmpframes' %path.basename(filename)
 
                # get the format of the movie
                format = getformat(filename)
                nrows = int(format['ID_VIDEO_HEIGHT'])
                ncols = int(format['ID_VIDEO_WIDTH'])
                seconds = float(format['ID_LENGTH'])
                framerate = float(format['ID_VIDEO_FPS'])
                nframes = int(seconds*framerate)
 
                self.colorspace = colorspace
 
                # get the number of color channels (so we can know the size of a
                # single frame)
                if colorspace in ['y800','y8']:
                        nchan = 1
                        bitdepth = 'uint8'
                        atomsize = 1
                elif colorspace in ['rgb16','rgb24','bgr16','bgr24']:
                        nchan = 3
                        bitdepth = 'uint8'
                        atomsize = 1
                elif colorspace in ['rgba','argb','rgb32','bgra','abgr','bgr32']:
                        nchan = 4
                        atomsize = 1
                        bitdepth = 'uint8'
                else:
                        raise Exception('Sorry - "%s" is not a currently supported colorspace',colorspace)
 
                self.blocking = blocking
                self.info = format
                self.filename = filename
                self.framerate = framerate
                self.duration = seconds
                self.shape = (nframes,nrows,ncols,nchan)
                self.bitdepth = bitdepth
 
                self._framesize = nrows*ncols*nchan*atomsize
                self._offsets = self._getoffsets()
                self._cache_p = cache
 
                # make temporary file containing the raw frames
                self.frames = self.decode()
 
        def __getitem__(self,key):
                """
                Grab frames from the source movie as numpy arrays
                """
 
                nframes,nrows,ncols,nchan = self.shape
                bits = self.bitdepth
                rcount = self._framesize
                offsets = self._offsets
                raw = self._cache_h
                thread = self._thread
 
                if thread.isAlive():
                        print 'Not done decoding frames yet'
                        return
 
                if raw.closed:
                        print 'Temporary frame cache is closed. Call .decode() to create a new frame cache.'
                        return
 
                if isinstance(key,tuple):
                        raise IndexError('Too many indices')
                elif isinstance(key,int):
                        # -ve indices read from the end
                        if key < 0:
                                key = nframes+key
                        indices = (key,)
                elif isinstance(key,slice):
                        indices = np.arange(*key.indices(nframes),dtype='uint16')
 
                outsize = (len(indices),nrows,ncols,nchan)
                framesout = np.empty(outsize,dtype=bits)
 
                for ii in xrange(len(indices)):
                        oo = offsets[indices[ii]]
                        raw.seek(oo,0)
                        frame = np.fromfile(raw,count=rcount,dtype=bits)
                        framesout[ii] = frame.reshape(self.shape[1:])
 
                return framesout.squeeze()
 
        def __iter__(self):
                """
                Iterate over the source movie, return frames as numpy arrays
                """
                nframes = self.shape[0]
                bits = self.bitdepth
                rcount = self._framesize
                raw = self._cache_h
                thread = self._thread
 
                if thread.isAlive():
                        print 'Not done decoding frames yet'
                        return
 
                if raw.closed:
                        print 'Temporary frame cache is closed. Call .decode() to create a new frame cache.'
                        return
 
                raw.seek(0,0)
                for ii in xrange(nframes):
                        frame = np.fromfile(raw,count=rcount,dtype=bits)
                        yield frame.reshape(self.shape[1:])
 
        def close(self):
                """
                Close the temporary frame cache and delete it from the disk
                """
                print "Deleting temporary frame cache"
                self._cache_h.close()
                remove(self._cache_p)
 
        def decode(self):
 
                # wipe any existing data in the cache
                file(self._cache_p,'w')
 
                # read the .avi, dump the output in a temporary file
                cmdstring = (   'mencoder',
                                self.filename,
                                '-ac','none',
                                '-ovc','raw',
                                '-nosound',
                                '-vf','format=%s' %self.colorspace,
                                '-of','rawvideo',
                                '-o',self._cache_p,
                                '-really-quiet'
                                )
 
                def _notifyOnExit(onExit, popenArgs,blocking):
                        def runInThread(onExit, popenArgs):
                                proc = subprocess.Popen(popenArgs[0],**popenArgs[1])
                                proc.wait()
                                onExit()
                                return
                        thread = threading.Thread(target=runInThread,args=(onExit,popenArgs))
                        thread.start()
                        if blocking:
                                # wait until thread completes
                                thread.join()
                        return thread
                def onExit():
                        print "Finished decoding frames in %s" %self.filename
                        return
 
                with open(devnull,"w") as fnull:
                        print "Decoding frames in %s to temporary cache. Be patient..." %self.filename
                        self._thread = _notifyOnExit(onExit,(cmdstring,dict(shell=False)),self.blocking)
                        # self._getframes = subprocess.Popen(cmdstring,stdout=fnull,stderr=fnull,shell=False)
 
                # open the temporary file for reading
                self._cache_h = file(self._cache_p,'r')
 
        def _getoffsets(self):
                offsets = []
                for index in xrange(self.shape[0]):
                        offsets.append(self._framesize*index)
                return offsets
 
 
def getformat(filename):
        """
        Grab the header information from a movie file using mplayer, return it
        as a dict
        """
 
        # use mplayer to grab the header information
        cmdstring = (   'mplayer',
                        '-vo','null',
                        '-ao','null',
                        '-frames','0',
                        '-really-quiet',
                        '-identify',
                        filename
                        )
 
        # suppress socket error messages
        with open(devnull, "w") as fnull:
                formatstr = subprocess.check_output(cmdstring,stderr=fnull,shell=False)
        lines = formatstr.splitlines()
        format = {}
 
        for line in lines:
                name,value = line.split('=')
                format.update({name:value})
 
        return format

if __name__ == "__main__":
        # test color...
        frames = np.random.random_integers(0,255,size=(50,100,200,3)).astype('uint8')
        vsnk = VideoSink('test_rgb.avi',size=frames.shape[1:3],colorspace='rgb24')
        for frame in frames:
                vsnk(frame)
        vsnk.close()
        # test grayscale...
        frames = np.random.random_integers(0,255,size=(1000,100,100,1)).astype('uint8')
        vsnk = VideoSink('test_gray.avi',size=frames.shape[1:3],colorspace='y8')
        for frame in frames:
                vsnk(frame)
        vsnk.close()
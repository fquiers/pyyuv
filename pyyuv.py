# A YUV file reader/writer

# Imports
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

# Constants
CSPACE_420 = 420
CSPACE_422 = 422
DEPTH_8  = 8
DEPTH_10 = 10

# YUV file handler class
class YuvFile:

    """A YUV file handler class"""

    def __init__(self, name, mode, width, height, colour, depth):
        """YuvFile class constructor"""
        self.name = name
        self.mode = mode
        self.width = width
        self.height = height
        self.colour = colour
        self.depth = depth
        self.handle = None
        # Chroma sub-sampling factor wrt. luma
        self.h_smpl = 2
        if self.colour == CSPACE_420:
            self.v_smpl = 2
        else:
            self.v_smpl = 1
        if self.depth == DEPTH_10:
            self.dtype = np.uint16
            self.round = 1
            self.shift = 2
        else:
            self.dtype = np.uint8
            self.round = 0
            self.shift = 0
        self.y_shape = (self.height, self.width)
        self.y_size = self.y_shape[0] * self.y_shape[1]
        self.uv_shape = (self.height / self.v_smpl, self.width / self.h_smpl)
        self.uv_size = self.uv_shape[0] * self.uv_shape[1]

    def open(self):
        """Opens the file associated with a YuvFile object"""
        self.handle = open(self.name, self.mode + "b")

    def close(self):
        """Closes the file associated with a YuvFile object"""
        self.handle.close()

    def __upsample_chroma(self, p):
        """Upsamples an array of chroma data from a 4:2:0 or 4:2:2 picture to 4:4:4"""
        q = np.kron(p, [1,1]) # Horizontal stretch
        if self.colour == CSPACE_420:
            q = np.kron(q, [[1],[1]]) # Vertical stretch
        return q

    def to_rgb_8b(self, y, u, v):
        """Converts a set of Y,U,V data to 8-bit RGB format"""
        cb = self.__upsample_chroma(u)
        cr = self.__upsample_chroma(v)
        y8 = np.right_shift(y + self.round, self.shift)
        cb8 = np.right_shift(cb + self.round, self.shift)
        cr8 = np.right_shift(cr + self.round, self.shift)
        r = np.clip(y8 + 1.402 * (cr8 - 2**7)                       , 0, 2**8 - 1)
        g = np.clip(y8 - 0.344 * (cb8 - 2**7) - 0.714 * (cr8 - 2**7), 0, 2**8 - 1)
        b = np.clip(y8 + 1.772 * (cb8 - 2**7)                       , 0, 2**8 - 1)
        return r,g,b

    def plot_luma(self, y):
        """Plots an array of luma data (gray scale)"""
        plt.imshow(y, cmap=plt.cm.gray)
        plt.show()

    def plot_rgb_8b(self, r, g, b):
        """Plots an 8-bit RGB picture"""
        rgb = np.swapaxes(np.swapaxes(np.array([r,g,b]).astype(np.uint8), 0, 2), 0, 1)
        plt.imshow(rgb)
        plt.show()

    def read_frame(self):
        """Reads one frame worth of Y,U,V data from file"""
        try:
            y = np.fromfile(self.handle, self.dtype, self.y_size).reshape(self.y_shape)
            u = np.fromfile(self.handle, self.dtype, self.uv_size).reshape(self.uv_shape)
            v = np.fromfile(self.handle, self.dtype, self.uv_size).reshape(self.uv_shape)
        except ValueError:
            y = np.empty([0, 0])
            u = np.empty([0, 0])
            v = np.empty([0, 0])
        return y,u,v

    def write_frame(self, y, u, v):
        """Writes one frame worth of Y,U,V data to file"""
        y.tofile(self.handle)
        u.tofile(self.handle)
        v.tofile(self.handle)

#######################################################################
#FIXME: FCQ: The code below is only used for testing during development
#Example: ./pyyuv.py -i input.yuv -w 1280 -t 720 -c 420 -d 8 -p 200 -o output.yuv

# Program options parser
def parse_options():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("-i", "--input",
                      dest="input", default=None,
                      help="Input file")
    parser.add_option("-o", "--output",
                      dest="output", default=None,
                      help="Output file (optional)")
    parser.add_option("-w", "--width", type="int",
                      dest="width", default=0,
                      help="Picture width in pixels - must be even")
    parser.add_option("-t", "--height", type="int",
                      dest="height", default=0,
                      help="Picture height in pixels - must be even")
    parser.add_option("-c", "--colour-space", type="int",
                      dest="colour", default=CSPACE_420,
                      help="Colour space: %d or %d (default: %d)" % (CSPACE_420, CSPACE_422, CSPACE_420))
    parser.add_option("-d", "--bit-depth", type="int",
                      dest="depth", default=DEPTH_8,
                      help="Bit depth: %d or %d (default: %d)" % (DEPTH_8, DEPTH_10, DEPTH_8))
    parser.add_option("-f", "--first-frame", type="int",
                      dest="first", default=0,
                      help="Index of first frame to process (default: 0)")
    parser.add_option("-n", "--num-frames", type="int",
                      dest="frames", default=None,
                      help="Number of frames to process (default: all)")
    parser.add_option("-p", "--plot-interval", type="int",
                      dest="plot", default=None,
                      help="Plot interval as a number of frames (default: no plot)")
    (opts, args) = parser.parse_args()
    if args:
        parser.print_help()
        parser.error("Unexpected arguments")
    if not opts.input:
        parser.print_help()
        parser.error("No input file name specified")
    if not os.path.isfile(opts.input):
        parser.print_help()
        parser.error("Input file not found: %s" % opts.input)
    if opts.width <= 0 or opts.height <= 0 or opts.width % 2 or opts.height % 2:
        parser.print_help()
        parser.error("Invalid width and/or height")
    if opts.colour not in [CSPACE_420, CSPACE_422]:
        parser.print_help()
        parser.error("Invalid colour space")
    if opts.depth not in [DEPTH_8, DEPTH_10]:
        parser.print_help()
        parser.error("Invalid bit depth")
    if (opts.first < 0):
        parser.print_help()
        parser.error("Invalid index for first frame")
    if (opts.frames != None) and opts.frames <= 0:
        parser.print_help()
        parser.error("Invalid number of frames")
    if opts.plot and opts.plot <= 0:
        parser.print_help()
        parser.error("Invalid plot interval")
    return opts

# Main function
def main():
    print "===== PyYuv test script ====="
    opts = parse_options()
    handle = YuvFile(opts.input, "r", opts.width, opts.height, opts.colour, opts.depth)
    ohandle = None
    if opts.output:
        ohandle = YuvFile(opts.output, "w", opts.width, opts.height, opts.colour, opts.depth)
    handle.open()
    if ohandle:
        ohandle.open()
    frame = 0
    while (not opts.frames) or (frame < opts.first + opts.frames):
        y,u,v = handle.read_frame()
        if not np.size(y):
            break
        if frame >= opts.first:
            print "[Frame %d]" % frame
            if opts.plot > 0 and (frame - opts.first) % opts.plot == 0:
                handle.plot_luma(y)
                r,g,b = handle.to_rgb_8b(y,u,v)
                handle.plot_rgb_8b(r,g,b)
            if ohandle:
                ohandle.write_frame(y,u,v)
        frame += 1
    handle.close()
    if ohandle:
        ohandle.close()
    print "Done."
    return 0

# Entry point
if __name__ == "__main__":
    sys.exit(main())

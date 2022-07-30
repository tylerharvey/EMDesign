'''
Script to split and SVG-ize individual elements (e.g. equipotentials, electrodes) of a MEBS HGPL plot.
Outputted SVGs can be read by Inkscape's import function.

Run as
$ python3 split_hpgl.py <hpgl_filename>
'''
from hpgl2svg import parse_file
import sys, os, re
try:
    infile = sys.argv[1]
except IndexError:
    print('HPGL filename must be provided as a command line argument.')
infile_noext, ext = os.path.splitext(infile)
ext = '.plt'
# ext = '.hpgl'
print(infile)
input_file = open(infile,'r')
outputs = re.split(r'SP\s+\d+;\s+PT\s+\d+\.?\d*;\s+',input_file.read())[1:]
for i,output in enumerate(outputs):
    print(f'File contents:\n{output[:50]}')
    output_filename = infile_noext+f'_{i}'+ext
    output_file = open(output_filename,'w+')
    output_file.write(output)
    output_file.close()
    tag = '_equipot' if i == 1 else ''
    svg_name = infile_noext+f'_{i}{tag}.svg'
    parse_file(output_filename,svg_name)


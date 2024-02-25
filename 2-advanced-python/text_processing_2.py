# -*- coding: utf-8 -*-
"""
Created on Thu May 25 08:24:27 2023

@author: sumit
"""

s = '   hello world     \n'
s.replace(' ','')

#========================
import re 
re.sub('\s+', ' ', s)
#' hello world '

s = 'pýtĥöñ\fis\tawesome\r\n'

s
#'pýtĥöñ\x0cis\tawesome\r\n'

#the first step is to clean up the whitespace to do this make a small translation
# and use translate 
remap = {
    ord('\t') : ' ',
    ord('\f') : None,
    ord('\r') : None
    }
a = s.translate(remap)
a
#'pýtĥöñis awesome\n'

remap = {
    ord('\t') : ' ',
    ord('\f') : ' ',
    ord('\r') : None
    }
a = s.translate(remap)
a
#'pýtĥöñ is awesome\n'

import sys
import unicodedata 
cmb_chrs = dict.fromkeys(c for c in range(sys.maxunicode) if unicodedata.combining(chr(c)))

b = unicodedata.normalize('NFD',a)
b
#'pýtĥöñ is awesome\n'

# Yet another technique for cleaning up text 
#Invloves I/O decoding and encoding functions 
# The idea here is to first do some preeliminary 
#cleanup of the text, and then run it 
#through a combination of encode() or dexode() operations
# to strip or alter it.
a = 'pýtĥöñ is awesome\n' 

b = unicodedata.normalize('NFD',a)
b.encode('ascii' , 'ignore').decode('ascii')

#=====================================================
#Aligning the text string 
text = 'Hello World'
text.ljust(20)
#'Hello World  
text.rjust(20)
#'         Hello World'
text.center(20) 
#'    Hello World     '

#===============================================
#All of thesse methods accept an optional charecters 7 fill charecter as well 
text.rjust(20,'=')
#'=========Hello World' 
text.center(20,'^')
#'^^^^Hello World^^^^^' 
#--------------------------------------------------
format(text,'>20')
#'         Hello World' 

format(text,'<20')
#'Hello World         '

format(text,'^20')
#'    Hello World     '



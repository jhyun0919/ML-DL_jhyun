#!/Users/JH/Documents/GitHub/ML&DL_jhyun/datascience/datascience/bin/python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/pathos/browser/dill/LICENSE

if __name__ == '__main__':
  import sys
  import dill
  for file in sys.argv[1:]:
    print (dill.load(open(file,'r')))


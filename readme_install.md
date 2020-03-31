### Python:

Make sure Python (3.6), pip, homebrew and virtualenv are installed.

$ python3 -m venv venv && . venv/bin/activate
$ pip install -r project_files/requirements.txt


### Gurobi:

1. Download Gurobi Optimizer3 and install in your computer. (You will need to register as an academic user, or purchase a license.)
http://user.gurobi.com/download/gurobi-optimizer

2. Request a free academic license4 and follow their instruction to activate it.
http://user.gurobi.com/download/licenses/free-academic

3. run command:
$ export GUROBI_HOME=/Library/gurobi811/mac64/lib

(Note to Windows users: The version you select, either 32-bit or 64-bit, needs to be consistent. That is, if you choose 64-bit Gurobi Optimizer, you will need to install 64-bit Julia in the next step. After installation, you must reboot your computer.)

### Julia: 

$ brew cask install julia 

Note: this command work only for mac, for windows follow https://www.softcover.io/read/7b8eb7d0/juliabook/introduction from 1.3.2 Installing Julia in Windows.  

$ sudo ln -s /Applications/Julia-1.3.app/Contents/Resources/julia/bin/julia /usr/local/bin/julia

$ julia --project=project_files/pomato
$ ] instantiate

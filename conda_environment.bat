@echo off

set envName=py37_Project3_Flask
set pyVersion=3.7

rem You must use "call" otherwise Windows stops executing after the first non-call 
call conda create --name %envName% python=%pyVersion% --verbose --yes

rem install dependencies
rem call conda install flask --name %envName% --verbose --yes

rem activate the environment
call activate %envName%

rem install pip dependencies (these are not hosted by conda)
call pip install flask==1.1.1
call pip install flask_restplus
call pip install pycaret
call pip install pickle
call pip install gunicorn

rem known issue in flask
rem call pip uninstall werkzeug --yes 
rem call pip install werkzeug==0.16.0

rem display the current environment
call conda info

rem display the installed packages
call conda -n %envName% -pkg-name "Flask"
call pip list

python app.py
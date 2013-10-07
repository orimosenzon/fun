echo off 

echo Running py2exe.. this will create a folder name 'dist' which will contain executables equivalent to the python scripts...
echo =====================================================
echo.  
echo.  
 
c:\python25\python setup.py py2exe 
echo.  
echo.  
echo.  
echo.  


echo Running NSIS's make.. this will create the installation file 
echo =====================================================  
echo.  
echo.  
  
"C:\Program Files\NSIS\makensis" esla.nsi
echo.  
echo.  
echo.  
echo.  
echo created file: esla_setup.exe. This is the installation file. It is a stand alone application that takes care of all the installation process
echo.  
echo.  

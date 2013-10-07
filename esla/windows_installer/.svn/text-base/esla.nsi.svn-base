    !define VERSION "9.10" 
    Name "Esla ${VERSION}"

    Icon "C:\esla\windows_installer\esla.ico" 

    Caption "Esla Caption" 
    BrandingText "orimosenzon.com" 
    OutFile "esla_setup.exe" 
      
    LicenseText "GPL V3" 
    LicenseData "license_gpl_v3.txt" 
      
    InstallDir "$PROGRAMFILES\esla" 
    DirText "You are about to install Esla in the following directory. If you prefer using a different one, please browse" 

    Page license 

    Page directory 

    Page instfiles 

    Section "Install" 
            SetOutPath $INSTDIR
            File /r "dist\*" 
	    File /r "vec_comb.ico" 
	    File /r "trans.ico" 

	    CreateDirectory "$SMPROGRAMS\Esla"
	    CreateShortCut "$SMPROGRAMS\Esla\trans.lnk" "$INSTDIR\trans.exe" "" "$INSTDIR\trans.ico" 0
	    CreateShortCut "$SMPROGRAMS\Esla\vec_comb.lnk" "$INSTDIR\vec_comb.exe" "" "$INSTDIR\vec_comb.ico" 0 "SW_SHOWMAXIMIZED"
	    CreateShortCut "$SMPROGRAMS\Esla\Uninstall.lnk" "$INSTDIR\Uninstall.exe"

	    WriteUninstaller $INSTDIR\Uninstall.exe
    SectionEnd 

    Section "Uninstall" 

    	    ; delete the items in the start menu 
	    RMDir /r "$SMPROGRAMS\Esla"

	    ; delete the installation folder itself 
    	    RMDir /r "$INSTDIR"
    SectionEnd 

; installer/sets_warp.iss
; Inno Setup 6 script for SETS-WARP Windows installer.
;
; Build manually:
;   iscc /DAppVersion=1.0b installer\sets_warp.iss
;
; Output: installer\dist\sets-warp-1.0b-setup.exe
;
; Built automatically by GitHub Actions on every release.

#ifndef AppVersion
  #define AppVersion "dev"
#endif

#define AppName      "SETS-WARP"
#define AppPublisher "SETS-WARP"
#define AppURL       "https://github.com/raman78/sets-warp"

[Setup]
; Keep this GUID fixed — Windows uses it to detect reinstalls / upgrades
AppId={{6F3A1D2E-4B7C-4E8F-9A01-2C3D4E5F6071}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases

; Install to %LOCALAPPDATA%\SETS-WARP — no administrator rights required
DefaultDirName={localappdata}\{#AppName}
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
AllowNoIcons=yes

Compression=lzma2/ultra
SolidCompression=yes
OutputDir=dist
OutputBaseFilename=sets-warp-{#AppVersion}-setup

; Require 64-bit Windows
ArchitecturesInstallIn64BitMode=x64compatible

WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
; All project files — skips git data, venv, portable Python, caches, build output
Source: "..\*";             DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: ".git,.venv,.python,.pycache,__pycache__,*.pyc,*.pyo,.config,installer\dist,dist"
Source: "..\installer\*";   DestDir: "{app}\installer"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}";          Filename: "{sys}\wscript.exe"; Parameters: """{app}\launch.vbs"""; WorkingDir: "{app}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#AppName}";  Filename: "{sys}\wscript.exe"; Parameters: """{app}\launch.vbs"""; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
; Offer to launch the app after installation
Filename: "{sys}\wscript.exe"; Parameters: """{app}\launch.vbs"""; WorkingDir: "{app}"; Description: "Launch {#AppName} now"; Flags: postinstall nowait skipifsilent

[Code]
// After install: check if Python 3.11+ is available and warn if not.
function IsPythonOK(): Boolean;
var
  ResultCode: Integer;
begin
  // Try Windows py launcher first (most reliable on Windows)
  Result := Exec(ExpandConstant('{sys}\py.exe'), '-3.11 --version', '',
                 SW_HIDE, ewWaitUntilTerminated, ResultCode)
            and (ResultCode = 0);
  if not Result then
    Result := Exec('python.exe',
                   '-c "import sys; exit(0 if sys.version_info>=(3,11) else 1)"', '',
                   SW_HIDE, ewWaitUntilTerminated, ResultCode)
              and (ResultCode = 0);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    if not IsPythonOK() then
      MsgBox(
        'Python 3.11 or later was not found on this computer.' + #13#10 + #13#10 +
        'Please install Python from:' + #13#10 +
        'https://www.python.org/downloads/' + #13#10 + #13#10 +
        'Important: check "Add Python to PATH" during installation.' + #13#10 +
        'Then launch SETS-WARP from the desktop shortcut.',
        mbInformation, MB_OK);
end;

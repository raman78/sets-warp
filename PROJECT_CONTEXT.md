Project: SETS-WARP

Purpose:
Star Trek Online game build creator.

SETS:
Main  toolf for build creation.

STO Equipment and Build Selector, called SETS, is a new and innovative planning tool for ship and ground builds as well as space and ground skill trees for Star Trek Online, developed by the STO Community Developers. Our main goal is to provide an easy and intuitive platform for optimizing every single aspect of a space or ground build, aimed at brand new and experienced players likewise. You can also save and share your builds as JavaScript Object Notation (.json) or Portable Network Graphic (.png) file and even an in markdown pre-formatted text export is included. Its unique connection to the STO Wiki almost guarantees an up-to-date database without the need of manual additions to the app itself.

Features
Plan space and ground builds on any ship.
Build without being restricted to owned items.
Share builds and edit shared builds (JSON and PNG format).
Convenient skill tree experience: SETS allows you to select and deselect skill points at will, which is not possible in game.
Export builds in markdown format.
Open the wiki page of any item using the context menu.

WARP:
WARP - Weaponry & Armament Recognition Platform
Tool for detecting Star Trek Online builds from screenshots, by using  ML models. Working locally, but with synchronization on web from users data.

Main assumpiotns:
1. SETS-WARP (SETS and WARP) is standalone program, independent from system modules, libraries; this is why it has autoconfigurator, which is installing modules, libraries and all needed components in .venv, using portable .python, updates and removes unnecessary dependencies.
2. SETS-WARP is using SyncManager to download, update data (images) from github and stowiki page (stowiki right now blocked by cloudflare anti-bot system)
3. All code comments and program communication messages in English

Main modules:
warp/recognition
warp/trainer
warp/knowledge

Technologies:
Python
OpenCV
PyTorch
ONNX

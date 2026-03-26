# Roadmap: SETS-WARP Separation

## Goal
Decouple WARP-specific logic from the original `src/` (SETS) directory. This allows for seamless updates from the `SETS` upstream repository while maintaining WARP features as a "plugin" or extension.

## Guiding Principles
*   **Zero Functionality Loss**: No additional SETS functionality and no WARP features and user settings must be lost during or after the separation.
*   **Clean `src/`**: The `src/` directory should eventually be an exact copy of the upstream `SETS` repository.
*   **Runtime Injection**: Use inheritance and monkey patching to add WARP features at runtime.

---

## Phase 0: Preparation & Analysis
* [x] **0.1 Setup Upstream Remote**: Add original SETS repo as `upstream`. (Assumed done or managed externally)
* [x] **0.2 Create Vendor Branch**: Create `vendor-sets` branch to track clean upstream. (User action required: `git checkout -b vendor-sets upstream/main`)
* [x] **0.3 Change Audit**: Identified all WARP modifications in `src/` within `sets-warp`.

## Phase 1: Establish WARP Entry Point
* [x] **1.1 Create `warp/app.py`**: Define `WarpSETS(SETS)` class inheriting from `src.app.SETS`. Moved WARP-specific logic (`_get_install_mode`, `_save_install_mode`, `_WARP_AVAILABLE`, app name overrides, UI injection logic, installer logic) from `src/app.py` to `warp/app.py`.
* [x] **1.2 Update `main.py`**: Switch entry point to `warp.app.WarpSETS`.
* [x] **1.3 Basic Overrides**: Moved version strings (`version`, `__version__`), application naming, and Windows Taskbar ID to `WarpSETS`.

## Phase 2: Migrate UI & Feature Injections
* [ ] **2.1 UI Helpers Migration**: Move menu injections, "Warp Core" section, and update checks to `warp/ui_helpers.py`.
* [ ] **2.2 Initialization Flow**: Call UI helpers from `WarpSETS.__init__` or a dedicated `_inject_warp_ui` method.
* [ ] **2.3 Cleanup `src/app.py`**: Remove all UI injection code from the original source.
* [ ] **2.4 Installer Logic**: Move WARP installation/uninstallation logic to the `warp` module.

## Phase 3: Monkey Patching & Source Cleanup
* [ ] **3.1 Identify Complex Modifications**: List modifications in `src/` modules other than `app.py`.
* [ ] **3.2 Implement Patches**: Use method overriding in `WarpSETS` or monkey patching in `warp/patches.py` for:
    * `src.buildupdater.update_boff_seat`
    * `src.widgetbuilder` UI overrides
    * Custom constant injections in `src.constants`.
* [ ] **3.3 Full Source Restore**: Revert all files in `src/` to their original `SETS` state.

## Phase 4: Verification & Upstream Integration
* [ ] **4.1 Regression Testing**: Verify all SETS and WARP features work post-separation.
* [ ] **4.2 Sync with Upstream**: Merge `upstream/main` into `main` branch.
* [ ] **4.3 Feature Integration**: Integrate "Science Destroyer" update (`ae63ef1`) into the clean `src/` directory.

---

## Technical Approach: Decoupling Strategy
1. **Inheritance**: `WarpSETS` extends `SETS` to override specific behaviors.
2. **Monkey Patching**: Dynamic runtime replacement of functions in `src` modules to add WARP logic without editing files.
3. **Dependency Injection**: UI elements are injected into layouts after they are initialized by the base class.

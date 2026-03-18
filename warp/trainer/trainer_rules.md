# STO TRAINER RULES

## 1. Item Name and Slot Type Compatibility
Each item is assigned a specific **Slot Type**. Items may only be equipped in slots matching their category, with the following mandatory logic:

* **Weapons**: Can ONLY be placed in **Fore Weapons** or **Aft Weapons** slots.
* **Experimental Weapons**: Can ONLY be placed in the **Experimental Weapons** slot.
* **Devices**: Can ONLY be placed in **Devices** slots.
* **Deflectors**: Can ONLY be placed in the **Deflector** slot.
* **Secondary Deflectors**: Can ONLY be placed in the **Sec-Def** slot.
* **Impulse Engines**: Can ONLY be placed in the **Engines** slot.
* **Warp Cores / Singularity Cores**: Can ONLY be placed in the **Warp Core** slot.
* **Shields**: Can ONLY be placed in the **Shield** slot.
* **Consoles (Logic)**:
    * **Universal Consoles**: Highly versatile. Can be placed in **ANY** of the following: 
        * Universal Consoles
        * Tactical Consoles
        * Engineering Consoles
        * Science Consoles
    * **Tactical Consoles (Standard)**: Restricted to **Tactical** or **Universal** slots.
    * **Engineering Consoles (Standard)**: Restricted to **Engineering** or **Universal** slots.
    * **Science Consoles (Standard)**: Restricted to **Science** or **Universal** slots.
* **Hangar Pets (Shuttles/Fighters)**: Can ONLY be placed in **Hangars** slots.
* **Ship Metadata**:
    * **Ship Name**: User-defined string. Positioned above Ship Type/Tier.
    * **Ship Type**: Must match the predefined list of Star Trek Online ship classes.
    * **Ship Tier**: Restricted to a dropdown selection: **T1-T6, T5-X, T5-U, T6-X, T6-X2**.
* **General Rule**: Any other **Slot Type** not explicitly listed above is restricted to its namesake slot (e.g., a "Kit Module" slot only accepts "Kit Module" items).

## 2. Dropdown Filtering Logic
The **Slot Type** acts as a primary filter for the **Item Name** search field. 
* When a user interacts with a specific slot, the search results must be programmatically limited to items that satisfy the compatibility rules defined in Section 1.

## 3. Screen Type Restrictions
The **Screen Type** determines the global availability of **Slot Types** within the interface:

| Screen Type | Slot Restriction |
| :--- | :--- |
| **Unknown** | No restrictions; all slot types may be selected/displayed. |
| **Space Mixed (merged)** | Restricted to **Ship Build** slots only (Weapons, Consoles, Space Gear, Traits, Devices, Boffs, Primary Specialization, Secondary Specialization). |
| **Ground Mixed (merged)** | Restricted to **Ground Build** slots only (Armor, Shields, Weapons, Kits, Modules, Traits, Boffs, Primary Specialization, Secondary Specialization). |

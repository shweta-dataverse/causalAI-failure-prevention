# causRCA preprocessing: data-quality report

## Policies

- **timestamps**: OPC UA publish times; ties broken by file row order, not interpreted causally
- **duplicates**: same (recording, node, time_s): keep last
- **state**: zero-order hold (forward fill) within a recording; NA before first event; no backfill
- **grid**: t_k = k * interval_s from 0 to the first grid point at or after the last event
- **categorical**: kept as strings, never converted to numbers

## Recordings

- Total: 270 (dig_twin: 100, real_op: 170)
- Fault runs per subsystem: coolant: 25, hydraulics: 41, probe: 34
- Duration (dig_twin): min 140.4 s, median 218.8 s, max 364.1 s
- Duration (real_op): min 80.1 s, median 599.5 s, max 613.7 s
- Naming notes (8): file name refers to exp2; folder exp_2_3 is used as experiment_id

## Events

- Events after duplicate policy: 508466
- Duplicate events: 0 (dropped 0)
- Out-of-order rows in files: 0
- Events (t > 0) sharing their timestamp with another node: 73.4%
- Re-published values without change: 79
- Binary/Alarm raw encodings: {'False': 53056, 'True': 53135}
- Numeric-looking categoricals kept as strings: Prog_LineNo
- Counter resets: {'Prog_CuttingTime': 9, 'Prog_CycleTime': 429}

## Nodes

- Distinct nodes: 96; types: {'Alarm': 18, 'Binary': 64, 'Categorical': 4, 'Continuous': 7, 'Counter': 3}
- Nodes without a t = 0 snapshot in some recording: none
- **dig_twin**: 90 nodes observed in 100 recordings
  - missing in some recordings: CLF_A_700307, CLF_Filter_Ok, CLT_A_700310, CLT_Level_lt_Min, ExU_On, ExU_isOff, F_A_700313, F_Filter_Ok, HP_A_700304, HP_Pump_Ok, Hyd_A_700202, Hyd_A_700203, Hyd_A_700204, Hyd_A_700205, Hyd_A_700206, Hyd_A_700207, Hyd_A_700208, Hyd_Filter_Ok, Hyd_IsEnabled, Hyd_Level_Ok, Hyd_Pressure, Hyd_Pump_Ok, Hyd_Pump_On, Hyd_Pump_isOff, Hyd_Temp_lt_70, Hyd_Temp_lt_80, Hyd_Valve_P_Up, LP_A_700301, LP_Pump_Ok, LT_A_700317, LT_Level_Ok, LT_Pump_Ok, Lubr_On, Lubr_P_Ok, MPA_A_701124, MPA_A_701125, M_ErrorActive, M_Lifebit, M_WarnActive, Prog_A_701330, SL_Yellow, SRL_Locked, SRL_Unlocked, SRL_lock, SRL_unlock, SR_tighten, TL_Locked
  - never change (0): none
- **real_op**: 80 nodes observed in 170 recordings
  - missing in some recordings: Prog_A_701330, SR_A_67040, T_A_701309, WC_collet_inPos, WC_finPart_inPos, WC_inTendingPos, WC_rawPart_inPos
  - never change (16): CLF_Filter_Ok, CLT_Level_lt_Min, F_Filter_Ok, HP_Pump_Ok, Hyd_Filter_Ok, Hyd_Level_Ok, Hyd_Pump_Ok, Hyd_Pump_On, Hyd_Pump_isOff, Hyd_Temp_lt_70, Hyd_Temp_lt_80, LP_Pump_Ok, WC_collet_inPos, WC_finPart_inPos, WC_inTendingPos, WC_rawPart_inPos

## State-table verification

- Interval 1.0 s, 270 recordings, 11668992 cells checked
- Mismatches: 0, grid errors: 0 -> **PASSED**

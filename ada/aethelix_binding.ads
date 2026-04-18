--  aethelix_binding.ads
--  Ada 2012 thin binding to the Aethelix flight diagnostic C API.
--
--  Target: LEON3 (GNAT for SPARC V8), RTEMS or bare-metal runtime.
--  Corresponds to: include/aethelix.h  (Rust ffi.rs)
--
--  Usage example (Ada 2012 FDIR task):
--
--    with Aethelix_Binding; use Aethelix_Binding;
--    with Interfaces.C;     use Interfaces.C;
--    with System;
--
--    procedure FDIR_Task is
--       State_Size : constant Unsigned := Aethelix_State_Size;
--       State_Mem  : aliased Byte_Array (1 .. Integer (State_Size))
--                       := (others => 0);
--       Alert      : aliased Aethelix_Alert;
--       RC         : Int;
--    begin
--       loop
--          RC := Process_Frame
--             (CCSDS_Buf  => Frame_Buffer'Address,
--              Buf_Len    => Frame_Buffer'Length,
--              Out_Alert  => Alert'Access,
--              State_Ptr  => State_Mem'Address);
--
--          if RC = AETHELIX_OK and then Alert.Level >= LEVEL_CAUTION then
--             Handle_Fault (Alert);
--          end if;
--       end loop;
--    end FDIR_Task;
--


with Interfaces.C;  use Interfaces.C;
with System;

package Aethelix_Binding is

   pragma Pure;

   
   VERSION_MAJOR : constant := 0;
   VERSION_MINOR : constant := 2;
   VERSION_PATCH : constant := 0;

  
   LEVEL_NONE     : constant Unsigned_Char := 0;
   LEVEL_WARNING  : constant Unsigned_Char := 1;
   LEVEL_CAUTION  : constant Unsigned_Char := 2;
   LEVEL_CRITICAL : constant Unsigned_Char := 3;


   AETHELIX_OK            : constant Int := 0;
   AETHELIX_ERR_TOO_SHORT : constant Int := 1;
   AETHELIX_ERR_BAD_LEN   : constant Int := 2;
   AETHELIX_ERR_NULL_PTR  : constant Int := -1;

 
   -- See include/aethelix_graph_ids.h for numeric values; these are
   -- the most common ones a FDIR task needs to branch on:
   FAULT_NONE                : constant Unsigned_Char := 16#FF#;
   FAULT_SOLAR_DEGRADATION   : constant Unsigned_Char := 16#00#;  -- approximate
   FAULT_BATTERY_AGING       : constant Unsigned_Char := 16#01#;
   FAULT_BATTERY_THERMAL     : constant Unsigned_Char := 16#02#;
   FAULT_SENSOR_BIAS         : constant Unsigned_Char := 16#03#;
   FAULT_PCDU_FAILURE        : constant Unsigned_Char := 16#04#;


   -- Must match AethelixAlert in include/aethelix.h (12 bytes, packed).
   type Aethelix_Alert is record
      Level             : Unsigned_Char;   --  LEVEL_* constant
      Root_Cause_Id     : Unsigned_Char;   --  FAULT_* or 16#FF# = none
      Confidence_Q15    : Short;           --  0–32767 (100 % = 32767)
      Evidence_Mask     : Unsigned;        --  Bitmask: bit N = channel N
      Onset_Frames_Ago  : Unsigned_Short;  --  Frames since first detection
      Root_Cause_2_Id   : Unsigned_Char;   --  Runner-up, 16#FF# if absent
      Reserved          : Unsigned_Char;   --  Padding - keep zero
   end record
     with
       Convention => C,
       Size       => 12 * System.Storage_Unit;

   
   type Byte_Array is array (Positive range <>) of Unsigned_Char
     with Convention => C;

 

   --  Return sizeof(AethelixState). Allocate at least this many bytes
   --  and zero-initialise before the first call to Process_Frame.
   function Aethelix_State_Size return Unsigned
     with Import        => True,
          Convention    => C,
          External_Name => "aethelix_state_size";

   --  Process one CCSDS Space Packet.
   --
   --  CCSDS_Buf : address of the raw packet byte array.
   --  Buf_Len   : number of valid bytes at CCSDS_Buf.
   --  Out_Alert : access to an Aethelix_Alert record (cleared on entry).
   --  State_Ptr : System.Address of the zeroed persistent state buffer.
   --
   --  Returns AETHELIX_OK on success, or one of the AETHELIX_ERR_* codes.
   function Process_Frame
     (CCSDS_Buf : System.Address;
      Buf_Len   : Unsigned_Short;
      Out_Alert : access Aethelix_Alert;
      State_Ptr : System.Address) return Int
     with Import        => True,
          Convention    => C,
          External_Name => "aethelix_process_frame";

   --  Zero-reset the engine state (call after a watchdog reset).
   procedure Reset_State (State_Ptr : System.Address)
     with Import        => True,
          Convention    => C,
          External_Name => "aethelix_reset_state";

   --  Recovery handler callback type
   type Recovery_Handler_Ptr is access procedure (Fault_Id : Int)
     with Convention => C;

   --  Register an active recovery handler callback for the FDIR framework.
   procedure Register_Recovery_Handler (Handler : Recovery_Handler_Ptr)
     with Import        => True,
          Convention    => C,
          External_Name => "register_recovery_handler";

end Aethelix_Binding;

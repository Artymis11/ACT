/* Shared pose selection: inactive status writes cannot animate stale feedback. */
(function(root) {
  'use strict';
  class TwinPoseState {
    constructor() { this.record = null; }
    update(data) {
      const valid = values => Array.isArray(values) && values.length===7 && values.every(Number.isFinite);
      if (!data || !valid(data.positions) || typeof data.session_id!=='string') return false;
      if (this.record && this.record.session_id===data.session_id && data.live!==true) return false;
      if (this.record && this.record.session_id===data.session_id && this.record.stamp===data.stamp) return false;
      this.record={...data,positions:[...data.positions],targets:valid(data.targets)?[...data.targets]:null};
      return true;
    }
  }
  if (typeof module!=='undefined' && module.exports) module.exports=TwinPoseState;
  else root.TwinPoseState=TwinPoseState;
})(typeof window!=='undefined'?window:globalThis);

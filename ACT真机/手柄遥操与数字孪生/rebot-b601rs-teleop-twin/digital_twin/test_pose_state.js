const assert=require('node:assert/strict');
const PoseState=require('./pose_state.js');
const sample=(value,live=true,sid='a',stamp=value)=>({session_id:sid,live,stamp,positions:Array(7).fill(value),targets:Array(7).fill(value+1)});
const state=new PoseState();
assert.equal(state.update(sample(1)),true);
for(const phase of ['fault','stale','ended']) {
  assert.equal(state.update({...sample(5,false),state:phase}),false);
  assert.equal(state.record.positions[0],1);
}
assert.equal(state.update(sample(2)),true); // recovered real feedback
assert.equal(state.record.positions[0],2);
assert.equal(state.update(sample(3,false,'new')),true); // initial last snapshot in a new session
assert.equal(state.update(sample(4,false,'new')),false);
assert.equal(state.record.positions[0],3);
assert.equal(state.update({...sample(4),positions:[NaN]}),false);
const fresh=sample(6);state.update(fresh);fresh.positions[0]=999;
assert.equal(state.record.positions[0],6); // stored pose is independent of mutable telemetry
console.log('Pose-state regression passed: stale/fault/ended freeze; live recovery; session switching; invalid input.');

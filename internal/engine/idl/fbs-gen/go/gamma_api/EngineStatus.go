// Code generated by the FlatBuffers compiler. DO NOT EDIT.

package gamma_api

import (
	flatbuffers "github.com/google/flatbuffers/go"
)

type EngineStatus struct {
	_tab flatbuffers.Table
}

func GetRootAsEngineStatus(buf []byte, offset flatbuffers.UOffsetT) *EngineStatus {
	n := flatbuffers.GetUOffsetT(buf[offset:])
	x := &EngineStatus{}
	x.Init(buf, n+offset)
	return x
}

func (rcv *EngineStatus) Init(buf []byte, i flatbuffers.UOffsetT) {
	rcv._tab.Bytes = buf
	rcv._tab.Pos = i
}

func (rcv *EngineStatus) Table() flatbuffers.Table {
	return rcv._tab
}

func (rcv *EngineStatus) IndexStatus() int32 {
	o := flatbuffers.UOffsetT(rcv._tab.Offset(4))
	if o != 0 {
		return rcv._tab.GetInt32(o + rcv._tab.Pos)
	}
	return 0
}

func (rcv *EngineStatus) MutateIndexStatus(n int32) bool {
	return rcv._tab.MutateInt32Slot(4, n)
}

func (rcv *EngineStatus) TableMem() int64 {
	o := flatbuffers.UOffsetT(rcv._tab.Offset(6))
	if o != 0 {
		return rcv._tab.GetInt64(o + rcv._tab.Pos)
	}
	return 0
}

func (rcv *EngineStatus) MutateTableMem(n int64) bool {
	return rcv._tab.MutateInt64Slot(6, n)
}

func (rcv *EngineStatus) IndexMem() int64 {
	o := flatbuffers.UOffsetT(rcv._tab.Offset(8))
	if o != 0 {
		return rcv._tab.GetInt64(o + rcv._tab.Pos)
	}
	return 0
}

func (rcv *EngineStatus) MutateIndexMem(n int64) bool {
	return rcv._tab.MutateInt64Slot(8, n)
}

func (rcv *EngineStatus) VectorMem() int64 {
	o := flatbuffers.UOffsetT(rcv._tab.Offset(10))
	if o != 0 {
		return rcv._tab.GetInt64(o + rcv._tab.Pos)
	}
	return 0
}

func (rcv *EngineStatus) MutateVectorMem(n int64) bool {
	return rcv._tab.MutateInt64Slot(10, n)
}

func (rcv *EngineStatus) FieldRangeMem() int64 {
	o := flatbuffers.UOffsetT(rcv._tab.Offset(12))
	if o != 0 {
		return rcv._tab.GetInt64(o + rcv._tab.Pos)
	}
	return 0
}

func (rcv *EngineStatus) MutateFieldRangeMem(n int64) bool {
	return rcv._tab.MutateInt64Slot(12, n)
}

func (rcv *EngineStatus) BitmapMem() int64 {
	o := flatbuffers.UOffsetT(rcv._tab.Offset(14))
	if o != 0 {
		return rcv._tab.GetInt64(o + rcv._tab.Pos)
	}
	return 0
}

func (rcv *EngineStatus) MutateBitmapMem(n int64) bool {
	return rcv._tab.MutateInt64Slot(14, n)
}

func (rcv *EngineStatus) DocNum() int32 {
	o := flatbuffers.UOffsetT(rcv._tab.Offset(16))
	if o != 0 {
		return rcv._tab.GetInt32(o + rcv._tab.Pos)
	}
	return 0
}

func (rcv *EngineStatus) MutateDocNum(n int32) bool {
	return rcv._tab.MutateInt32Slot(16, n)
}

func (rcv *EngineStatus) MaxDocid() int32 {
	o := flatbuffers.UOffsetT(rcv._tab.Offset(18))
	if o != 0 {
		return rcv._tab.GetInt32(o + rcv._tab.Pos)
	}
	return 0
}

func (rcv *EngineStatus) MutateMaxDocid(n int32) bool {
	return rcv._tab.MutateInt32Slot(18, n)
}

func (rcv *EngineStatus) MinIndexedNum() int32 {
	o := flatbuffers.UOffsetT(rcv._tab.Offset(20))
	if o != 0 {
		return rcv._tab.GetInt32(o + rcv._tab.Pos)
	}
	return 0
}

func (rcv *EngineStatus) MutateMinIndexedNum(n int32) bool {
	return rcv._tab.MutateInt32Slot(20, n)
}

func EngineStatusStart(builder *flatbuffers.Builder) {
	builder.StartObject(9)
}
func EngineStatusAddIndexStatus(builder *flatbuffers.Builder, indexStatus int32) {
	builder.PrependInt32Slot(0, indexStatus, 0)
}
func EngineStatusAddTableMem(builder *flatbuffers.Builder, tableMem int64) {
	builder.PrependInt64Slot(1, tableMem, 0)
}
func EngineStatusAddIndexMem(builder *flatbuffers.Builder, indexMem int64) {
	builder.PrependInt64Slot(2, indexMem, 0)
}
func EngineStatusAddVectorMem(builder *flatbuffers.Builder, vectorMem int64) {
	builder.PrependInt64Slot(3, vectorMem, 0)
}
func EngineStatusAddFieldRangeMem(builder *flatbuffers.Builder, fieldRangeMem int64) {
	builder.PrependInt64Slot(4, fieldRangeMem, 0)
}
func EngineStatusAddBitmapMem(builder *flatbuffers.Builder, bitmapMem int64) {
	builder.PrependInt64Slot(5, bitmapMem, 0)
}
func EngineStatusAddDocNum(builder *flatbuffers.Builder, docNum int32) {
	builder.PrependInt32Slot(6, docNum, 0)
}
func EngineStatusAddMaxDocid(builder *flatbuffers.Builder, maxDocid int32) {
	builder.PrependInt32Slot(7, maxDocid, 0)
}
func EngineStatusAddMinIndexedNum(builder *flatbuffers.Builder, minIndexedNum int32) {
	builder.PrependInt32Slot(8, minIndexedNum, 0)
}
func EngineStatusEnd(builder *flatbuffers.Builder) flatbuffers.UOffsetT {
	return builder.EndObject()
}
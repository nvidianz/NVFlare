# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: federated.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0f\x66\x65\x64\x65rated.proto\x12\x1cxgboost.collective.federated\"N\n\x10\x41llgatherRequest\x12\x17\n\x0fsequence_number\x18\x01 \x01(\x04\x12\x0c\n\x04rank\x18\x02 \x01(\x05\x12\x13\n\x0bsend_buffer\x18\x03 \x01(\x0c\"(\n\x0e\x41llgatherReply\x12\x16\n\x0ereceive_buffer\x18\x01 \x01(\x0c\"O\n\x11\x41llgatherVRequest\x12\x17\n\x0fsequence_number\x18\x01 \x01(\x04\x12\x0c\n\x04rank\x18\x02 \x01(\x05\x12\x13\n\x0bsend_buffer\x18\x03 \x01(\x0c\")\n\x0f\x41llgatherVReply\x12\x16\n\x0ereceive_buffer\x18\x01 \x01(\x0c\"\xd2\x01\n\x10\x41llreduceRequest\x12\x17\n\x0fsequence_number\x18\x01 \x01(\x04\x12\x0c\n\x04rank\x18\x02 \x01(\x05\x12\x13\n\x0bsend_buffer\x18\x03 \x01(\x0c\x12\x39\n\tdata_type\x18\x04 \x01(\x0e\x32&.xgboost.collective.federated.DataType\x12G\n\x10reduce_operation\x18\x05 \x01(\x0e\x32-.xgboost.collective.federated.ReduceOperation\"(\n\x0e\x41llreduceReply\x12\x16\n\x0ereceive_buffer\x18\x01 \x01(\x0c\"\\\n\x10\x42roadcastRequest\x12\x17\n\x0fsequence_number\x18\x01 \x01(\x04\x12\x0c\n\x04rank\x18\x02 \x01(\x05\x12\x13\n\x0bsend_buffer\x18\x03 \x01(\x0c\x12\x0c\n\x04root\x18\x04 \x01(\x05\"(\n\x0e\x42roadcastReply\x12\x16\n\x0ereceive_buffer\x18\x01 \x01(\x0c*\x96\x01\n\x08\x44\x61taType\x12\x08\n\x04HALF\x10\x00\x12\t\n\x05\x46LOAT\x10\x01\x12\n\n\x06\x44OUBLE\x10\x02\x12\x0f\n\x0bLONG_DOUBLE\x10\x03\x12\x08\n\x04INT8\x10\x04\x12\t\n\x05INT16\x10\x05\x12\t\n\x05INT32\x10\x06\x12\t\n\x05INT64\x10\x07\x12\t\n\x05UINT8\x10\x08\x12\n\n\x06UINT16\x10\t\x12\n\n\x06UINT32\x10\n\x12\n\n\x06UINT64\x10\x0b*^\n\x0fReduceOperation\x12\x07\n\x03MAX\x10\x00\x12\x07\n\x03MIN\x10\x01\x12\x07\n\x03SUM\x10\x02\x12\x0f\n\x0b\x42ITWISE_AND\x10\x03\x12\x0e\n\nBITWISE_OR\x10\x04\x12\x0f\n\x0b\x42ITWISE_XOR\x10\x05\x32\xc2\x03\n\tFederated\x12k\n\tAllgather\x12..xgboost.collective.federated.AllgatherRequest\x1a,.xgboost.collective.federated.AllgatherReply\"\x00\x12n\n\nAllgatherV\x12/.xgboost.collective.federated.AllgatherVRequest\x1a-.xgboost.collective.federated.AllgatherVReply\"\x00\x12k\n\tAllreduce\x12..xgboost.collective.federated.AllreduceRequest\x1a,.xgboost.collective.federated.AllreduceReply\"\x00\x12k\n\tBroadcast\x12..xgboost.collective.federated.BroadcastRequest\x1a,.xgboost.collective.federated.BroadcastReply\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'federated_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_DATATYPE']._serialized_start=687
  _globals['_DATATYPE']._serialized_end=837
  _globals['_REDUCEOPERATION']._serialized_start=839
  _globals['_REDUCEOPERATION']._serialized_end=933
  _globals['_ALLGATHERREQUEST']._serialized_start=49
  _globals['_ALLGATHERREQUEST']._serialized_end=127
  _globals['_ALLGATHERREPLY']._serialized_start=129
  _globals['_ALLGATHERREPLY']._serialized_end=169
  _globals['_ALLGATHERVREQUEST']._serialized_start=171
  _globals['_ALLGATHERVREQUEST']._serialized_end=250
  _globals['_ALLGATHERVREPLY']._serialized_start=252
  _globals['_ALLGATHERVREPLY']._serialized_end=293
  _globals['_ALLREDUCEREQUEST']._serialized_start=296
  _globals['_ALLREDUCEREQUEST']._serialized_end=506
  _globals['_ALLREDUCEREPLY']._serialized_start=508
  _globals['_ALLREDUCEREPLY']._serialized_end=548
  _globals['_BROADCASTREQUEST']._serialized_start=550
  _globals['_BROADCASTREQUEST']._serialized_end=642
  _globals['_BROADCASTREPLY']._serialized_start=644
  _globals['_BROADCASTREPLY']._serialized_end=684
  _globals['_FEDERATED']._serialized_start=936
  _globals['_FEDERATED']._serialized_end=1386
# @@protoc_insertion_point(module_scope)

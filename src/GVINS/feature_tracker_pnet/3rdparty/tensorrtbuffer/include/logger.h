/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LOGGER_H
#define LOGGER_H

#include "logging.h"

class TensorRTErrorRecorder;
extern TensorRTErrorRecorder gRecorder;
namespace tensorrt_log
{
    extern Logger gLogger;
    extern LogStreamConsumer gLogVerbose;
    extern LogStreamConsumer gLogInfo;
    extern LogStreamConsumer gLogWarning;
    extern LogStreamConsumer gLogError;
    extern LogStreamConsumer gLogFatal;

    void setReportableSeverity(Logger::Severity severity);
} // namespace sample

#endif // LOGGER_H
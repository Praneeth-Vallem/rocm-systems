/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <sys/sysinfo.h>

#include "suites/functional/time_stamp.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

#define RET_IF_HSA_ERR(err) { \
  if ((err) != HSA_STATUS_SUCCESS) { \
    const char* msg = 0; \
    hsa_status_string(err, &msg); \
    std::cout << "hsa api call failure at line " << __LINE__ << ", file: " << \
                          __FILE__ << ". Call returned " << err << std::endl; \
    std::cout << msg << std::endl; \
    return (err); \
  } \
}


TimeStamp::TimeStamp(void) :
    TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  set_title("RocR verify clock counter");
  set_description("This series of tests captures 2 sets of  timestamps from KFD,"
    " to get snapshots of CPU vs GPU ticks; confirming that the tickets coming from KFD are correct.");
}

TimeStamp::~TimeStamp(void) {
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void TimeStamp::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  return;
}

void TimeStamp::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void TimeStamp::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TimeStamp::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void TimeStamp::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

static const char kSubTestSeparator[] = "  **************************";

static void PrintMemorySubtestHeader(const char *header) {
  std::cout << "  *** Memory Subtest: " << header << " ***" << std::endl;
}

void TimeStamp::TimeStampTest (void) {
  hsa_status_t err;
  std::vector<std::shared_ptr<rocrtst::agent_pools_t>> agent_pools;
  char ag_name[64];
  hsa_device_type_t ag_type;
  PrintMemorySubtestHeader("verifing clock counter IOCTLs");

  err = rocrtst::GetAgentPools(&agent_pools);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  auto pool_idx = 0;
   std::cout<<"Total number of agent_pools: "<<agent_pools.size()<<std::endl;
  for (auto a : agent_pools) {
        hsa_amd_clock_counters_t counter = {};
        err = hsa_agent_get_info(a->agent, HSA_AGENT_INFO_NAME, ag_name);
        ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  
        err = hsa_agent_get_info(a->agent, HSA_AGENT_INFO_DEVICE, &ag_type);
        ASSERT_EQ(err, HSA_STATUS_SUCCESS);
        if (verbosity() > 0) {
            std::cout << std::endl<<"  Agent: " << ag_name << " (";
            switch (ag_type) {
              case HSA_DEVICE_TYPE_CPU:
                std::cout << "CPU)";
                break;
              case HSA_DEVICE_TYPE_GPU:
                std::cout << "GPU)";
                break;
              case HSA_DEVICE_TYPE_DSP:
                std::cout << "DSP)";
                break;
              case HSA_DEVICE_TYPE_AIE:
                std::cout << "AIE)";
                break;
        }
          std::cout << std::endl;
        }
        ASSERT_EQ(hsa_agent_get_info(a->agent, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_CLOCK_COUNTERS,
                            &counter), HSA_STATUS_SUCCESS);

        std::cout<<" gpu_clock_counter: "<<counter.gpu_clock_counter <<std::endl;
        std::cout<<" cpu_clock_counter: "<<counter.cpu_clock_counter <<std::endl;
        std::cout<<" system_clock_counter: "<<counter.system_clock_counter <<std::endl;
        std::cout<<" system_clock_frequency: "<<counter.system_clock_frequency <<std::endl;
  }
}

 


#undef RET_IF_HSA_ERR

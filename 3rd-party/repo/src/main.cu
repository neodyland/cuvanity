﻿
#include <iostream>
#include <cstdint>
#include <random>
#include <chrono>
#include <thread>
#include <atomic>
#include <set>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "keccak.h"
#define EXPAND 10
#define EXPANDBUF 4096
#include "gpu_keccak.h"
#include "gpu_crypto.h"

constexpr size_t PATTERN_SIZE = 32;
#include "gpu_scan.h"

extern "C" {
#include "crypto-ops.h"
}

using namespace std::chrono;

constexpr char alphabet32[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

constexpr size_t BATCH_SIZE = 1 << 20;

int main(int argc, char** argv)
{
#define CHECKED_CALL(X) do { \
        const cudaError_t err = X; \
        if (err != cudaSuccess) { \
            std::cerr << #X " (line " << __LINE__ << ") failed, error " << err; \
            return __LINE__; \
        } \
    } while(0)

    int device_count;
    CHECKED_CALL(cudaGetDeviceCount(&device_count));
    
    bool rate_info = false;
    
    std::string patterns_str;

    std::set<int> devices_to_use;

    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];

	// enable hashrate reporting
	if (s == "-i") {
	  rate_info = true;
	  continue;
	}
	
	// device selection
        if ((s == "-d") && (i + 1 < argc)) {
            ++i;
            const int id = strtol(argv[i], nullptr, 10);
            if (0 <= id && id < device_count) {
                devices_to_use.insert(id);
            }
            else {
                printf("Invalid device id %s\n", argv[i]);
            }
            continue;
        }

        if (s.length() > PATTERN_SIZE) {
            s.resize(PATTERN_SIZE);
        }
        else {
            while (s.length() < PATTERN_SIZE) {
                s += '?';
            }
        }

	
        const char* abc = alphabet32;

        bool good = true;
        for (int j = 0; j < PATTERN_SIZE; ++j) {
            if (s[j] == '?') {
                continue;
            }
            bool found = false;
            for (int k = 0; k < 58; ++k) {
                if (s[j] == abc[k]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                good = false;
                break;
            }
        }
        if (good) {
            patterns_str += s;
        }
        else {
            std::cout << "Invalid pattern \"" << argv[i] << "\"" << std::endl;
        }
    }

    if (patterns_str.empty()) {
        printf(
            "Usage:\n\n"
            "./vanity_torv3_cuda [-d N] pattern1 [pattern_2] [pattern_3] ... [pattern_n]\n\n"
	    "-i         will print out the hashrate every 20 seconds if flag set\n\n"
            "-d N       use CUDA device with index N (counting from 0). This argument can be repeated multiple times with different N.\n\n"
            "Each pattern can have \"?\" symbols which match any character.\nOnly the following characters are allowed:\n\n abcdefghijklmnopqrstuvwxyz234567\n\n"
            "Example:\n\t./vanity_torv3_cuda P?XXXXX L23456 b55555 FfFfFf H99999\n\n"
            "If the vanity generator finds a match, it will print the secret key as it's running.\n"
            "These can be appended by 32bytes of secure randomness to make a torv3 key file.\n\n"
        );
        return 0;
    }
    
    // Make all uppercase 
    for (int i = 0; patterns_str[i]!='\0'; i++) {
      if(patterns_str[i] >= 'a' && patterns_str[i] <= 'z') {
	patterns_str[i] = patterns_str[i] -32;
      }
    }
    
    // Get some entropy from the random device
    std::random_device::result_type rnd_buf[256];
    std::random_device rd;
    for (int i = 0; i < 256; ++i) {
        rnd_buf[i] = rd();
    }

    std::atomic<uint64_t> keys_checked;
    std::vector<std::thread> threads;

    for (int i = 0; i < device_count; ++i) {
        if (!devices_to_use.empty() && (devices_to_use.find(i) == devices_to_use.end())) {
            continue;
        }

        cudaDeviceProp prop;
        CHECKED_CALL(cudaGetDeviceProperties(&prop, i));
	std::cout << "Using CUDA device " << i << ": " << prop.name << std::endl;

        threads.emplace_back([i, &rnd_buf, &patterns_str, &keys_checked]()
        {
            CHECKED_CALL(cudaSetDevice(i));

            CHECKED_CALL(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

            // Mix entropy into 32-byte secret key template
            uint8_t tmp_buf[sizeof(rnd_buf)];
            memcpy(tmp_buf, rnd_buf, sizeof(rnd_buf));

            // Mix in thread number
            tmp_buf[0] ^= i;

            // Mix all bits of the random buffer into the key template
            uint8_t key_template[32];
            keccak(tmp_buf, sizeof(tmp_buf), key_template, sizeof(key_template), 24);

            uint8_t* input_buf;
            CHECKED_CALL(cudaMalloc((void**)&input_buf, 32));
            CHECKED_CALL(cudaMemcpy(input_buf, key_template, sizeof(key_template), cudaMemcpyHostToDevice));

            uint64_t* data;
            CHECKED_CALL(cudaMalloc((void**)&data, BATCH_SIZE * 32));
	    
            uint64_t* kdata;
            CHECKED_CALL(cudaMalloc((void**)&kdata, BATCH_SIZE * EXPANDBUF));

            uint8_t* patterns;
            CHECKED_CALL(cudaMalloc((void**)&patterns, patterns_str.length()));
            CHECKED_CALL(cudaMemcpy(patterns, patterns_str.data(), patterns_str.length(), cudaMemcpyHostToDevice));

            uint32_t* results_key;
            CHECKED_CALL(cudaMalloc((void**)&results_key, 256 * sizeof(uint32_t)));
	    uint32_t* results_ctr;
            CHECKED_CALL(cudaMalloc((void**)&results_ctr, 256 * sizeof(uint32_t)));

            for (uint64_t offset = 0;; offset += BATCH_SIZE) {
                CHECKED_CALL(cudaMemset(results_key, 0, sizeof(uint32_t)));

                gpu::keccak_12_rounds<<<BATCH_SIZE, 32>>>(input_buf, 32, offset, data);
                CHECKED_CALL(cudaGetLastError());
		
                gpu::reduce<<<BATCH_SIZE / 32, 32>>>(data);
                CHECKED_CALL(cudaGetLastError());
		
                gpu::gen_public_keys_primary<<<BATCH_SIZE / 32, 32>>>(data, kdata);
                CHECKED_CALL(cudaGetLastError());

		// TODO: add in a conditional to switch to this kernel the majority of the time once Primary has run at least once
		//gpu::gen_public_keys_secondary<<<BATCH_SIZE / 32, 32>>>(data, kdata);
                //CHECKED_CALL(cudaGetLastError());

                gpu::prefix_scan<<<BATCH_SIZE / 32, 32>>>(patterns, patterns_str.length() / PATTERN_SIZE, kdata, results_key, results_ctr);
                CHECKED_CALL(cudaGetLastError());

                CHECKED_CALL(cudaDeviceSynchronize());

                uint32_t results_key_host[256];
                CHECKED_CALL(cudaMemcpy(results_key_host, results_key, sizeof(results_key_host), cudaMemcpyDeviceToHost));
                uint32_t results_ctr_host[256];
                CHECKED_CALL(cudaMemcpy(results_ctr_host, results_ctr, sizeof(results_ctr_host), cudaMemcpyDeviceToHost));
		
		uint32_t counter = 0;
                for (uint32_t i = 1, n = std::min(255u, results_key_host[0]); i <= n; ++i) {
                    uint8_t buf[32];
		    counter = results_ctr_host[i];

		    *((uint64_t*)key_template) ^= offset + results_key_host[i];
		    keccak(key_template, 32, buf, 32, 12);
		    *((uint64_t*)key_template) ^= offset + results_key_host[i];

                    sc_reduce32(buf);

                    uint8_t tor_secret_key[32];
		    uint8_t tor_secret_key_[32];
		    
                    memcpy(tor_secret_key, buf, 32);
		    uint8_t fixup[32];
		    memset(fixup, 0, 32);
		    
		    uint32_t incr = (counter+1) * 8;
		    memcpy(fixup, &incr, sizeof(incr));
		      
		    scalar_add(tor_secret_key_,tor_secret_key, fixup);

                    uint8_t encoded_tor_secret_key[65];
                    for (int j = 0; j < 32; ++j) {
                        encoded_tor_secret_key[j * 2    ] = "0123456789abcdef"[tor_secret_key_[j] >> 4];
                        encoded_tor_secret_key[j * 2 + 1] = "0123456789abcdef"[tor_secret_key_[j] & 15];
		    }
                    encoded_tor_secret_key[64] = '\0';
		    
		    //std::cout << encoded_tor_secret_key << " " << n << " " << i << std::endl;
		    // probably logic error on my part, but will filter out 99% of false-positives vs speed
		    if(n == i) {
		      std::cout << encoded_tor_secret_key << std::endl;
		    }
		    
                }

                keys_checked += BATCH_SIZE; // for every key we generate we expand it
            }
        });
    }

    auto t1 = high_resolution_clock::now();
    // Timing information in million keys generated per second.
    uint64_t prev_keys_checked = 0;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(20));

        const uint64_t cur_keys_checked = keys_checked;
        const auto t2 = high_resolution_clock::now();

        const double dt = duration_cast<nanoseconds>(t2 - t1).count() * 1e-9;
	if(rate_info)
	  std::cout << "KEYRATE: " << (cur_keys_checked - prev_keys_checked) / dt * 1e-6 << " million keys/second" << std::endl; // This value is not validated experimentally, but is a ballpark. 
	  // Protocol: Generating large amount of keys in a fixed amount of time had the point addition generate 10-15% more valid keys. TODO: fix this at same time as tuning loop.
        t1 = t2;
        prev_keys_checked = cur_keys_checked;
    }

    return 0;
}

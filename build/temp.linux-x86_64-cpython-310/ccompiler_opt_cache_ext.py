# AUTOGENERATED DON'T EDIT
# Please make changes to the code generator             (distutils/ccompiler_opt.py)
hash = 684305541
data = \
{'cache_infile': True,
 'cache_me': {"('cc_test_flags', ['-O3'])": True,
              "('cc_test_flags', ['-Werror'])": True,
              "('cc_test_flags', ['-march=native'])": True,
              "('cc_test_flags', ['-mavx'])": True,
              "('cc_test_flags', ['-mavx2'])": True,
              "('cc_test_flags', ['-mavx5124fmaps', '-mavx5124vnniw', '-mavx512vpopcntdq'])": True,
              "('cc_test_flags', ['-mavx512cd'])": True,
              "('cc_test_flags', ['-mavx512er', '-mavx512pf'])": True,
              "('cc_test_flags', ['-mavx512f', '-mno-mmx'])": True,
              "('cc_test_flags', ['-mavx512ifma', '-mavx512vbmi'])": True,
              "('cc_test_flags', ['-mavx512vbmi2', '-mavx512bitalg', '-mavx512vpopcntdq'])": True,
              "('cc_test_flags', ['-mavx512vl', '-mavx512bw', '-mavx512dq'])": True,
              "('cc_test_flags', ['-mavx512vnni'])": True,
              "('cc_test_flags', ['-mf16c'])": True,
              "('cc_test_flags', ['-mfma'])": True,
              "('cc_test_flags', ['-mpopcnt'])": True,
              "('cc_test_flags', ['-msse'])": True,
              "('cc_test_flags', ['-msse2'])": True,
              "('cc_test_flags', ['-msse3'])": True,
              "('cc_test_flags', ['-msse4.1'])": True,
              "('cc_test_flags', ['-msse4.2'])": True,
              "('cc_test_flags', ['-mssse3'])": True,
              "('feature_extra_checks', 'AVX')": [],
              "('feature_extra_checks', 'AVX2')": [],
              "('feature_extra_checks', 'AVX512CD')": [],
              "('feature_extra_checks', 'AVX512F')": ['AVX512F_REDUCE'],
              "('feature_extra_checks', 'AVX512_CLX')": [],
              "('feature_extra_checks', 'AVX512_CNL')": [],
              "('feature_extra_checks', 'AVX512_ICL')": [],
              "('feature_extra_checks', 'AVX512_KNL')": [],
              "('feature_extra_checks', 'AVX512_KNM')": [],
              "('feature_extra_checks', 'AVX512_SKX')": ['AVX512BW_MASK',
                                                         'AVX512DQ_MASK'],
              "('feature_extra_checks', 'F16C')": [],
              "('feature_extra_checks', 'FMA3')": [],
              "('feature_extra_checks', 'POPCNT')": [],
              "('feature_extra_checks', 'SSE')": [],
              "('feature_extra_checks', 'SSE2')": [],
              "('feature_extra_checks', 'SSE3')": [],
              "('feature_extra_checks', 'SSE41')": [],
              "('feature_extra_checks', 'SSE42')": [],
              "('feature_extra_checks', 'SSSE3')": [],
              "('feature_flags', 'AVX')": ['-msse', '-msse2', '-msse3',
                                           '-mssse3', '-msse4.1', '-mpopcnt',
                                           '-msse4.2', '-mavx'],
              "('feature_flags', 'AVX2')": ['-msse', '-msse2', '-msse3',
                                            '-mssse3', '-msse4.1', '-mpopcnt',
                                            '-msse4.2', '-mavx', '-mf16c',
                                            '-mavx2'],
              "('feature_flags', 'AVX512CD')": ['-msse', '-msse2', '-msse3',
                                                '-mssse3', '-msse4.1',
                                                '-mpopcnt', '-msse4.2', '-mavx',
                                                '-mf16c', '-mfma', '-mavx2',
                                                '-mavx512f', '-mno-mmx',
                                                '-mavx512cd'],
              "('feature_flags', 'AVX512F')": ['-msse', '-msse2', '-msse3',
                                               '-mssse3', '-msse4.1',
                                               '-mpopcnt', '-msse4.2', '-mavx',
                                               '-mf16c', '-mfma', '-mavx2',
                                               '-mavx512f', '-mno-mmx'],
              "('feature_flags', 'AVX512_CLX')": ['-msse', '-msse2', '-msse3',
                                                  '-mssse3', '-msse4.1',
                                                  '-mpopcnt', '-msse4.2',
                                                  '-mavx', '-mf16c', '-mfma',
                                                  '-mavx2', '-mavx512f',
                                                  '-mno-mmx', '-mavx512cd',
                                                  '-mavx512vl', '-mavx512bw',
                                                  '-mavx512dq',
                                                  '-mavx512vnni'],
              "('feature_flags', 'AVX512_CNL')": ['-msse', '-msse2', '-msse3',
                                                  '-mssse3', '-msse4.1',
                                                  '-mpopcnt', '-msse4.2',
                                                  '-mavx', '-mf16c', '-mfma',
                                                  '-mavx2', '-mavx512f',
                                                  '-mno-mmx', '-mavx512cd',
                                                  '-mavx512vl', '-mavx512bw',
                                                  '-mavx512dq', '-mavx512ifma',
                                                  '-mavx512vbmi'],
              "('feature_flags', 'AVX512_ICL')": ['-msse', '-msse2', '-msse3',
                                                  '-mssse3', '-msse4.1',
                                                  '-mpopcnt', '-msse4.2',
                                                  '-mavx', '-mf16c', '-mfma',
                                                  '-mavx2', '-mavx512f',
                                                  '-mno-mmx', '-mavx512cd',
                                                  '-mavx512vl', '-mavx512bw',
                                                  '-mavx512dq', '-mavx512vnni',
                                                  '-mavx512ifma',
                                                  '-mavx512vbmi',
                                                  '-mavx512vbmi2',
                                                  '-mavx512bitalg',
                                                  '-mavx512vpopcntdq'],
              "('feature_flags', 'AVX512_KNL')": ['-msse', '-msse2', '-msse3',
                                                  '-mssse3', '-msse4.1',
                                                  '-mpopcnt', '-msse4.2',
                                                  '-mavx', '-mf16c', '-mfma',
                                                  '-mavx2', '-mavx512f',
                                                  '-mno-mmx', '-mavx512cd',
                                                  '-mavx512er', '-mavx512pf'],
              "('feature_flags', 'AVX512_KNM')": ['-msse', '-msse2', '-msse3',
                                                  '-mssse3', '-msse4.1',
                                                  '-mpopcnt', '-msse4.2',
                                                  '-mavx', '-mf16c', '-mfma',
                                                  '-mavx2', '-mavx512f',
                                                  '-mno-mmx', '-mavx512cd',
                                                  '-mavx512er', '-mavx512pf',
                                                  '-mavx5124fmaps',
                                                  '-mavx5124vnniw',
                                                  '-mavx512vpopcntdq'],
              "('feature_flags', 'AVX512_SKX')": ['-msse', '-msse2', '-msse3',
                                                  '-mssse3', '-msse4.1',
                                                  '-mpopcnt', '-msse4.2',
                                                  '-mavx', '-mf16c', '-mfma',
                                                  '-mavx2', '-mavx512f',
                                                  '-mno-mmx', '-mavx512cd',
                                                  '-mavx512vl', '-mavx512bw',
                                                  '-mavx512dq'],
              "('feature_flags', 'F16C')": ['-msse', '-msse2', '-msse3',
                                            '-mssse3', '-msse4.1', '-mpopcnt',
                                            '-msse4.2', '-mavx', '-mf16c'],
              "('feature_flags', 'FMA3')": ['-msse', '-msse2', '-msse3',
                                            '-mssse3', '-msse4.1', '-mpopcnt',
                                            '-msse4.2', '-mavx', '-mf16c',
                                            '-mfma'],
              "('feature_flags', 'POPCNT')": ['-msse', '-msse2', '-msse3',
                                              '-mssse3', '-msse4.1',
                                              '-mpopcnt'],
              "('feature_flags', 'SSE')": ['-msse', '-msse2'],
              "('feature_flags', 'SSE2')": ['-msse', '-msse2'],
              "('feature_flags', 'SSE3')": ['-msse', '-msse2', '-msse3'],
              "('feature_flags', 'SSE41')": ['-msse', '-msse2', '-msse3',
                                             '-mssse3', '-msse4.1'],
              "('feature_flags', 'SSE42')": ['-msse', '-msse2', '-msse3',
                                             '-mssse3', '-msse4.1', '-mpopcnt',
                                             '-msse4.2'],
              "('feature_flags', 'SSSE3')": ['-msse', '-msse2', '-msse3',
                                             '-mssse3'],
              "('feature_flags', {'SSE', 'SSE2', 'SSE3'})": ['-msse', '-msse2',
                                                             '-msse3'],
              "('feature_is_supported', 'AVX', 'force_flags', 'macros', None, [])": True,
              "('feature_is_supported', 'AVX2', 'force_flags', 'macros', None, [])": True,
              "('feature_is_supported', 'AVX512CD', 'force_flags', 'macros', None, [])": True,
              "('feature_is_supported', 'AVX512F', 'force_flags', 'macros', None, [])": True,
              "('feature_is_supported', 'AVX512_CLX', 'force_flags', 'macros', None, [])": True,
              "('feature_is_supported', 'AVX512_CNL', 'force_flags', 'macros', None, [])": True,
              "('feature_is_supported', 'AVX512_ICL', 'force_flags', 'macros', None, [])": True,
              "('feature_is_supported', 'AVX512_KNL', 'force_flags', 'macros', None, [])": True,
              "('feature_is_supported', 'AVX512_KNM', 'force_flags', 'macros', None, [])": True,
              "('feature_is_supported', 'AVX512_SKX', 'force_flags', 'macros', None, [])": True,
              "('feature_is_supported', 'F16C', 'force_flags', 'macros', None, [])": True,
              "('feature_is_supported', 'FMA3', 'force_flags', 'macros', None, [])": True,
              "('feature_is_supported', 'POPCNT', 'force_flags', 'macros', None, [])": True,
              "('feature_is_supported', 'SSE', 'force_flags', 'macros', None, [])": True,
              "('feature_is_supported', 'SSE2', 'force_flags', 'macros', None, [])": True,
              "('feature_is_supported', 'SSE3', 'force_flags', 'macros', None, [])": True,
              "('feature_is_supported', 'SSE41', 'force_flags', 'macros', None, [])": True,
              "('feature_is_supported', 'SSE42', 'force_flags', 'macros', None, [])": True,
              "('feature_is_supported', 'SSSE3', 'force_flags', 'macros', None, [])": True,
              "('feature_test', 'AVX', None, 'macros', [])": True,
              "('feature_test', 'AVX2', None, 'macros', [])": True,
              "('feature_test', 'AVX512CD', None, 'macros', [])": True,
              "('feature_test', 'AVX512F', None, 'macros', [])": True,
              "('feature_test', 'AVX512_CLX', None, 'macros', [])": True,
              "('feature_test', 'AVX512_CNL', None, 'macros', [])": True,
              "('feature_test', 'AVX512_ICL', None, 'macros', [])": True,
              "('feature_test', 'AVX512_KNL', None, 'macros', [])": True,
              "('feature_test', 'AVX512_KNM', None, 'macros', [])": True,
              "('feature_test', 'AVX512_SKX', None, 'macros', [])": True,
              "('feature_test', 'F16C', None, 'macros', [])": True,
              "('feature_test', 'FMA3', None, 'macros', [])": True,
              "('feature_test', 'POPCNT', None, 'macros', [])": True,
              "('feature_test', 'SSE', None, 'macros', [])": True,
              "('feature_test', 'SSE2', None, 'macros', [])": True,
              "('feature_test', 'SSE3', None, 'macros', [])": True,
              "('feature_test', 'SSE41', None, 'macros', [])": True,
              "('feature_test', 'SSE42', None, 'macros', [])": True,
              "('feature_test', 'SSSE3', None, 'macros', [])": True},
 'cache_private': {'sources_status'},
 'cc_flags': {'native': ['-march=native'],
              'opt': ['-O3'],
              'werror': ['-Werror']},
 'cc_has_debug': False,
 'cc_has_native': False,
 'cc_is_cached': True,
 'cc_is_clang': False,
 'cc_is_gcc': True,
 'cc_is_icc': False,
 'cc_is_iccw': False,
 'cc_is_msvc': False,
 'cc_is_nocc': False,
 'cc_march': 'x64',
 'cc_name': 'gcc',
 'cc_noopt': False,
 'cc_on_aarch64': False,
 'cc_on_armhf': False,
 'cc_on_noarch': False,
 'cc_on_ppc64': False,
 'cc_on_ppc64le': False,
 'cc_on_s390x': False,
 'cc_on_x64': True,
 'cc_on_x86': False,
 'feature_is_cached': True,
 'feature_min': {'SSE', 'SSE3', 'SSE2'},
 'feature_supported': {'AVX': {'flags': ['-mavx'],
                               'headers': ['immintrin.h'],
                               'implies': ['SSE42'],
                               'implies_detect': False,
                               'interest': 8},
                       'AVX2': {'flags': ['-mavx2'],
                                'implies': ['F16C'],
                                'interest': 13},
                       'AVX512CD': {'flags': ['-mavx512cd'],
                                    'implies': ['AVX512F'],
                                    'interest': 21},
                       'AVX512F': {'extra_checks': ['AVX512F_REDUCE'],
                                   'flags': ['-mavx512f', '-mno-mmx'],
                                   'implies': ['FMA3', 'AVX2'],
                                   'implies_detect': False,
                                   'interest': 20},
                       'AVX512_CLX': {'detect': ['AVX512_CLX'],
                                      'flags': ['-mavx512vnni'],
                                      'group': ['AVX512VNNI'],
                                      'implies': ['AVX512_SKX'],
                                      'interest': 43},
                       'AVX512_CNL': {'detect': ['AVX512_CNL'],
                                      'flags': ['-mavx512ifma', '-mavx512vbmi'],
                                      'group': ['AVX512IFMA', 'AVX512VBMI'],
                                      'implies': ['AVX512_SKX'],
                                      'implies_detect': False,
                                      'interest': 44},
                       'AVX512_ICL': {'detect': ['AVX512_ICL'],
                                      'flags': ['-mavx512vbmi2',
                                                '-mavx512bitalg',
                                                '-mavx512vpopcntdq'],
                                      'group': ['AVX512VBMI2', 'AVX512BITALG',
                                                'AVX512VPOPCNTDQ'],
                                      'implies': ['AVX512_CLX', 'AVX512_CNL'],
                                      'implies_detect': False,
                                      'interest': 45},
                       'AVX512_KNL': {'detect': ['AVX512_KNL'],
                                      'flags': ['-mavx512er', '-mavx512pf'],
                                      'group': ['AVX512ER', 'AVX512PF'],
                                      'implies': ['AVX512CD'],
                                      'implies_detect': False,
                                      'interest': 40},
                       'AVX512_KNM': {'detect': ['AVX512_KNM'],
                                      'flags': ['-mavx5124fmaps',
                                                '-mavx5124vnniw',
                                                '-mavx512vpopcntdq'],
                                      'group': ['AVX5124FMAPS', 'AVX5124VNNIW',
                                                'AVX512VPOPCNTDQ'],
                                      'implies': ['AVX512_KNL'],
                                      'implies_detect': False,
                                      'interest': 41},
                       'AVX512_SKX': {'detect': ['AVX512_SKX'],
                                      'extra_checks': ['AVX512BW_MASK',
                                                       'AVX512DQ_MASK'],
                                      'flags': ['-mavx512vl', '-mavx512bw',
                                                '-mavx512dq'],
                                      'group': ['AVX512VL', 'AVX512BW',
                                                'AVX512DQ', 'AVX512BW_MASK',
                                                'AVX512DQ_MASK'],
                                      'implies': ['AVX512CD'],
                                      'implies_detect': False,
                                      'interest': 42},
                       'F16C': {'flags': ['-mf16c'],
                                'implies': ['AVX'],
                                'interest': 11},
                       'FMA3': {'flags': ['-mfma'],
                                'implies': ['F16C'],
                                'interest': 12},
                       'FMA4': {'flags': ['-mfma4'],
                                'headers': ['x86intrin.h'],
                                'implies': ['AVX'],
                                'interest': 10},
                       'POPCNT': {'flags': ['-mpopcnt'],
                                  'headers': ['popcntintrin.h'],
                                  'implies': ['SSE41'],
                                  'interest': 6},
                       'SSE': {'flags': ['-msse'],
                               'headers': ['xmmintrin.h'],
                               'implies': ['SSE2'],
                               'interest': 1},
                       'SSE2': {'flags': ['-msse2'],
                                'headers': ['emmintrin.h'],
                                'implies': ['SSE'],
                                'interest': 2},
                       'SSE3': {'flags': ['-msse3'],
                                'headers': ['pmmintrin.h'],
                                'implies': ['SSE2'],
                                'interest': 3},
                       'SSE41': {'flags': ['-msse4.1'],
                                 'headers': ['smmintrin.h'],
                                 'implies': ['SSSE3'],
                                 'interest': 5},
                       'SSE42': {'flags': ['-msse4.2'],
                                 'implies': ['POPCNT'],
                                 'interest': 7},
                       'SSSE3': {'flags': ['-mssse3'],
                                 'headers': ['tmmintrin.h'],
                                 'implies': ['SSE3'],
                                 'interest': 4},
                       'XOP': {'flags': ['-mxop'],
                               'headers': ['x86intrin.h'],
                               'implies': ['AVX'],
                               'interest': 9}},
 'hit_cache': True,
 'parse_baseline_flags': ['-msse', '-msse2', '-msse3'],
 'parse_baseline_names': ['SSE', 'SSE2', 'SSE3'],
 'parse_dispatch_names': ['SSSE3', 'SSE41', 'POPCNT', 'SSE42', 'AVX', 'F16C',
                          'FMA3', 'AVX2', 'AVX512F', 'AVX512CD', 'AVX512_KNL',
                          'AVX512_KNM', 'AVX512_SKX', 'AVX512_CLX',
                          'AVX512_CNL', 'AVX512_ICL'],
 'parse_is_cached': True,
 'parse_target_groups': {'SIMD_TEST': (True,
                                       ['AVX512_SKX', 'AVX512F',
                                        ('FMA3', 'AVX2'), 'SSE42'],
                                       [])},
 'sources_status': {}}
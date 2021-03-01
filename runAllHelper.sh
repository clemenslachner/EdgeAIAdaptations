#!/bin/bash

function runGPUInitial {
echo "Running GPU test_mask_detector to generate baselines (if not already present)"
python3 test_mask_detector.py --videoFolder videos --generateBaseline --outputFolder results_gpu_init --gpu 0 --useTf
echo ""
echo "Running GPU test_mask_detector to perform the actual test (overwrites old testRun.json)"
python3 test_mask_detector.py -n 30 -n 15 -w 300 -w 150 -w 0 --gray False --gray True --videoFolder videos  --outputFolder results_gpu_init --gpu 0 --useTf
echo ""
echo "Running GPU analyse to perform the analysis (overwrites old .pngs)"
python3 analyse.py --sourceFolder results_gpu_init
echo ""
echo "GPU Done"
}

function runTfInitial {
echo "Running Tf test_mask_detector to generate baselines (if not already present)"
python3 test_mask_detector.py --videoFolder videos --generateBaseline --outputFolder results_tf_init --useTf
echo ""
echo "Running Tf test_mask_detector to perform the actual test (overwrites old testRun.json)"
python3 test_mask_detector.py -n 30 -n 15 -w 300 -w 150 -w 0 --gray False --gray True --videoFolder videos  --outputFolder results_tf_init --useTf
echo ""
echo "Running Tf analyse to perform the analysis (overwrites old .pngs)"
python3 analyse.py --sourceFolder results_tf_init
echo ""
echo "Tf Done"
}

function runGPUVersion {
echo "Running GPU test_mask_detector to generate baselines (if not already present)"
python3 test_mask_detector.py --videoFolder videos --generateBaseline --outputFolder results_gpu --gpu 0 --useTf
echo ""
echo "Running GPU test_mask_detector to perform the actual test (overwrites old testRun.json)"
python3 test_mask_detector.py -n 10 -w 300 -w 0 --videoFolder videos  --outputFolder results_gpu --gpu 0 --useTf
echo ""
echo "Running GPU analyse to perform the analysis (overwrites old .pngs)"
python3 analyse.py --sourceFolder results_gpu
echo ""
echo "GPU Done"
}

function runTfVersion {
echo "Running Tf test_mask_detector to generate a baseline (if not already present)"
python3 test_mask_detector.py --videoFolder videos --generateBaseline --outputFolder results_tf_baseline --useTf --onlyFaces True
echo ""
echo "Running Tf test_mask_detector to perform a test with frame skipping adaptations (overwrites old testRun.json)"
python3 test_mask_detector.py -n 5 --videoFolder videos  --outputFolder results_tf_frameskipping --useTf --onlyFaces True
echo ""
echo "Running Tf test_mask_detector to perform a test with resizing adaptations (overwrites old testRun.json)"
python3 test_mask_detector.py -w 300 -w 200 --videoFolder videos --outputFolder results_tf_resizing --useTf --onlyFaces True
echo ""
echo "Running Tf test_mask_detector to perform a test with greyscaling adaptations (overwrites old testRun.json)"
python3 test_mask_detector.py --gray True --videoFolder videos --outputFolder results_tf_greyscaling --useTf --onlyFaces True
echo ""
echo "Running Tf test_mask_detector to perform a test with anonymization adaptations (overwrites old testRun.json)"
python3 test_mask_detector.py -a 0 -a 1 -a 2 -a 3 --videoFolder videos --outputFolder results_tf_anonymization --useTf --onlyFaces True
echo "copy singlePerson baseline into each test specific folder"
cp results_tf_baseline/singlePerson.mp4/baseline.json results_tf_frameskipping/singlePerson.mp4/
cp results_tf_baseline/singlePerson.mp4/baseline.json results_tf_resizing/singlePerson.mp4/
cp results_tf_baseline/singlePerson.mp4/baseline.json results_tf_greyscaling/singlePerson.mp4/
cp results_tf_baseline/singlePerson.mp4/baseline.json results_tf_anonymization/singlePerson.mp4/
echo "copy 9people baseline into each test specific folder"
cp results_tf_baseline/9people.mp4/baseline.json results_tf_frameskipping/9people.mp4/
cp results_tf_baseline/9people.mp4/baseline.json results_tf_resizing/9people.mp4/
cp results_tf_baseline/9people.mp4/baseline.json results_tf_greyscaling/9people.mp4/
cp results_tf_baseline/9people.mp4/baseline.json results_tf_anonymization/9people.mp4/
echo "copy schoolStrike baseline into each test specific folder"
cp results_tf_baseline/schoolStrike.mp4/baseline.json results_tf_frameskipping/schoolStrike.mp4/
cp results_tf_baseline/schoolStrike.mp4/baseline.json results_tf_resizing/schoolStrike.mp4/
cp results_tf_baseline/schoolStrike.mp4/baseline.json results_tf_greyscaling/schoolStrike.mp4/
cp results_tf_baseline/schoolStrike.mp4/baseline.json results_tf_anonymization/schoolStrike.mp4/
echo "Running Tf analyse to perform the analysis (overwrites old .pngs)"
echo "Anaylizing Frameskipping"
python3 analyse.py --sourceFolder results_tf_frameskipping
echo "Anaylizing Resizing"
python3 analyse.py --sourceFolder results_tf_resizing
echo "Anaylizing Greyscaling"
python3 analyse.py --sourceFolder results_tf_greyscaling
echo "Anaylizing Anonymization"
python3 analyse.py --sourceFolder results_tf_anonymization
echo "---"
echo "Tf Done"
}

function runtTfVersionQuickCheck {
echo "Running Tf test_mask_detector to generate a baseline (if not already present)"
python3 test_mask_detector.py --videoFolder videos --generateBaseline --outputFolder results_tf_quick --useTf --onlyFaces True
echo ""
echo "Running Tf test_mask_detector to perform a test with anonymization adaptations (overwrites old testRun.json)"
python3 test_mask_detector.py -a 2 --videoFolder videos --outputFolder results_tf_quick --useTf --onlyFaces True
echo "Running Tf analyse to perform the analysis (overwrites old .pngs)"
echo "Anaylizing Frameskipping"
python3 analyse.py --sourceFolder results_tf_quick
echo "---"
echo "Tf Done"
}

#runGPUInitial
#runTfInitial
#runGPUVersion
#runTfVersion
runtTfVersionQuickCheck

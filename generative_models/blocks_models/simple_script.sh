#!/bin/bash
echo "RUNNING SCRIPT"
tar -czf imod_bikld_results.tar *.pkl
aws s3 cp imod_bikld_results.tar s3://nipsmodels/imod_bikld_results.tar
echo "FINISHED SCRIPT"
date '+%A %W %Y %X'

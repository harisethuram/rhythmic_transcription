# cancel all jobs between job ids a and b
a=$1
b=$2
for ((jobid=a; jobid<=b; jobid++)); do
    scancel $jobid
done
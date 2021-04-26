#!/bin/bash
# Script to run multiple simulations 5 by 5 until the desired number is reached

helpFunction()
{
   echo ""
   echo "Usage: $0 -n num_sims"
   echo -e "\t-n Number of simulations to run (default = 30)"
   echo -e "\t-c Net configuration file (mandatory)"
   echo -e "\t-s Number of simulation steps (default = 60000)"
   echo -e "\t-w Number of steps for simulation to wait before learning (default = 3000)"
   echo -e "\t-r C2I communication success rate (in case -u is true, 100 must be multiple of this value) (default = 100)"
   echo -e "\t-u Flag to use comunication success rate as a step"
   echo -e "\t-b Step gap to recalate betweenness (default == 1000)"
   exit 1
}

run()
{
   network=$1
   step=$2
   wait=$3
   s_rate=$4
   b_gap=$5

   mult_step=5
   sleep_step=2
   if [ $num_sims -lt 5 ]
   then 
      mult_step=$num_sims
   fi

   succ_rate=$(bc -l <<<"scale=2;$s_rate/100")
   echo ""
   echo "Starting to run simulations with communication success rate $s_rate%" 
   for j in $(eval echo "{$mult_step..$num_sims..$mult_step}")
   do  
      sleep_time=0
      for i in $(eval echo "{1..$mult_step..1}")
      do
         sleep $sleep_time && python3 main.py -c $network -s $step -w $wait -r $succ_rate -b $b_gap > /dev/null 2>&1 &
         let "sleep_time+=sleep_step"
      done
      wait
      now=$(date +"%d/%m/%Y - %H:%M")
      echo -e "Finished running $j \t simulations with $s_rate% C2I success rate at \t $now"
   done
}

#Defining default values
num_sims=30
steps=60000
wait_learn=3000
rate=100
use_step=0
btw_gap=1000

# Get parameters from args
while getopts "n:c:s:w:r:b:u" opt
do
   case "$opt" in
      n ) num_sims="$OPTARG" ;;
      c ) net_file="$OPTARG" ;;
      s ) steps="$OPTARG" ;;
      w ) wait_learn="$OPTARG" ;;
      r ) rate="$OPTARG" ;;
      b ) btw_gap="$OPTARG" ;;
      u ) use_step=1 ;;
      * ) helpFunction ;;
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$net_file" ]
then
   echo "Network file information is mandatory!!";
   helpFunction
fi

if ((100 % $rate != 0)) && (($use_step == 1))
then
   echo "Parameter -r value not allowed"
   helpFunction
fi

# Begin script in case all parameters are correct
now=$(date +"%d/%m/%Y - %H:%M")

if (($use_step == 1))
then
   total=$((100 / $rate + 1))
   total=$(($total * $num_sims))
   echo "Script will run $total simulations in total"
   echo "Starting simulations in background..."
   echo -e "Starting time at \t\t\t $now"
   for i in $(eval echo "{0..100..$rate}")
   do
      run $net_file $steps $wait_learn $i $btw_gap
   done
else
   echo "Script will run $num_sims simulations in total"
   echo "Starting simulations in background..."
   echo -e "Starting time at \t\t\t $now"
   run $net_file $steps $wait_learn $rate $btw_gap
fi
echo ""
echo "Done"

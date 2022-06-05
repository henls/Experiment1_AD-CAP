/*
 * CloudSim Plus: A modern, highly-extensible and easier-to-use Framework for
 * Modeling and Simulation of Cloud Computing Infrastructures and Services.
 * http://cloudsimplus.org
 *
 *     Copyright (C) 2015-2021 Universidade da Beira Interior (UBI, Portugal) and
 *     the Instituto Federal de Educação Ciência e Tecnologia do Tocantins (IFTO, Brazil).
 *
 *     This file is part of CloudSim Plus.
 *
 *     CloudSim Plus is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     CloudSim Plus is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with CloudSim Plus. If not, see <http://www.gnu.org/licenses/>.
 */
package org.cloudsimplus.examples.autoscaling;

import org.cloudbus.cloudsim.allocationpolicies.VmAllocationPolicySimple;
import org.cloudbus.cloudsim.brokers.DatacenterBroker;
import org.cloudbus.cloudsim.brokers.DatacenterBrokerSimple;
import org.cloudbus.cloudsim.cloudlets.Cloudlet;
import org.cloudbus.cloudsim.cloudlets.CloudletSimple;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.Simulation;
import org.cloudbus.cloudsim.datacenters.Datacenter;
import org.cloudbus.cloudsim.datacenters.DatacenterSimple;
import org.cloudbus.cloudsim.hosts.Host;
import org.cloudbus.cloudsim.hosts.HostSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.ResourceProvisionerSimple;
import org.cloudbus.cloudsim.resources.Pe;
import org.cloudbus.cloudsim.resources.PeSimple;
import org.cloudbus.cloudsim.resources.Processor;
import org.cloudbus.cloudsim.resources.Ram;
import org.cloudbus.cloudsim.schedulers.cloudlet.CloudletSchedulerTimeShared;
import org.cloudbus.cloudsim.schedulers.vm.VmSchedulerTimeShared;
import org.cloudbus.cloudsim.utilizationmodels.UtilizationModel;
import org.cloudbus.cloudsim.utilizationmodels.UtilizationModelDynamic;
import org.cloudbus.cloudsim.utilizationmodels.UtilizationModelFull;
import org.cloudbus.cloudsim.utilizationmodels.UtilizationModelStochastic;
import org.cloudbus.cloudsim.utilizationmodels.UtilizationModelPlanetLab;
import org.cloudbus.cloudsim.vms.Vm;
import org.cloudbus.cloudsim.vms.VmSimple;
import org.cloudsimplus.autoscaling.HorizontalVmScaling;
import org.cloudsimplus.autoscaling.VerticalVmScaling;
import org.cloudsimplus.autoscaling.VerticalVmScalingSimple;
import org.cloudsimplus.autoscaling.resources.ResourceScaling;
import org.cloudsimplus.autoscaling.resources.ResourceScalingGradual;
import org.cloudsimplus.autoscaling.resources.ResourceScalingInstantaneous;
import org.cloudsimplus.builders.tables.CloudletsTableBuilder;
import org.cloudsimplus.listeners.EventInfo;
import org.cloudsimplus.listeners.EventListener;
import static org.cloudbus.cloudsim.utilizationmodels.UtilizationModel.Unit;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import java.lang.*;

import static java.util.Comparator.comparingDouble;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;

import org.cloudsimplus.examples.autoscaling.AnomalyModel;
import org.cloudsimplus.examples.autoscaling.SlaStatistic;

/**
 * An example that scales VM PEs up or down, according to the arrival of Cloudlets.
 * A {@link VerticalVmScaling}
 * is set to each {@link #createListOfScalableVms(int) initially created VM}.
 * Every VM will check at {@link #SCHEDULING_INTERVAL specific time intervals}
 * if its PEs {@link #upperCpuUtilizationThreshold(Vm) are over or underloaded},
 * according to a <b>static computed utilization threshold</b>.
 * Then it requests such PEs to be up or down scaled.
 *
 * <p>The example uses the CloudSim Plus {@link EventListener} feature
 * to enable monitoring the simulation and dynamically create objects such as Cloudlets and VMs at runtime.
 * It relies on
 * <a href="https://docs.oracle.com/javase/tutorial/java/javaOO/methodreferences.html">Java 8 Method References</a>
 * to set a method to be called for {@link Simulation#addOnClockTickListener(EventListener) onClockTick events}.
 * It enables getting notifications when the simulation clock advances, then creating and submitting new cloudlets.
 * </p>
 *
 * @author Manoel Campos da Silva Filho
 * @since CloudSim Plus 1.2.0
 * @see VerticalVmRamScalingExample
 * @see VerticalVmCpuScalingDynamicThreshold
 */
public class VerticalVmCpuScalingExampleRevised {
    /**
     * The interval in which the Datacenter will schedule events.
     * As lower is this interval, sooner the processing of VMs and Cloudlets
     * is updated and you will get more notifications about the simulation execution.
     * However, it can affect the simulation performance.
     *
     * <p>For this example, a large schedule interval such as 15 will make that just
     * at every 15 seconds the processing of VMs is updated. If a VM is overloaded, just
     * after this time the creation of a new one will be requested
     * by the VM's {@link HorizontalVmScaling Horizontal Scaling} mechanism.</p>
     *
     * <p>If this interval is defined using a small value, you may get
     * more dynamically created VMs than expected. Accordingly, this value
     * has to be trade-off.
     * For more details, see {@link Datacenter#getSchedulingInterval()}.</p>
    */
    private static final int SCHEDULING_INTERVAL = 1;
    private static final int HOSTS = 1;

    private static final int HOST_PES = 32;
    private static final int VMS = 1;
    private static final int VM_PES = 6;
    private static final int VM_RAM = 1200;
    private final CloudSim simulation;
    private DatacenterBroker broker0;
    private List<Host> hostList;
    private List<Vm> vmList;
    private List<Cloudlet> cloudletList;

    //private static final int CLOUDLETS = 6; sampled
    private static final int CLOUDLETS = 30;
    //private static final int CLOUDLETS_INITIAL_LENGTH = 60_000_000;sampled
    private static final int CLOUDLETS_INITIAL_LENGTH = 1_000; 
    private static final int CLOUDLETS_LENGTH_LOWER = 2_000;
    private static final int CLOUDLETS_LENGTH_UPPER = 5_000;

    private static final String workdir = System.getProperty("user.dir") + "/capacityPlanning/simulation/manoelcampos-cloudsimplus-cc58449/cloudsim-plus-examples/src/main/";

    private static final String[] TRACE_FILE_CPU = {"workload/my-trace/4665896876_28_cpu_0.1038", 
                                                    "workload/my-trace/4665896876_43_cpu_0.1038",
                                                    "workload/my-trace/4665896876_95_cpu_0.1038",
                                                    "workload/my-trace/4665896876_107_cpu_0.1038",
                                                    "workload/my-trace/4665896876_173_cpu_0.1038",
                                                    "workload/my-trace/4665896876_183_cpu_0.1038",
                                                    "workload/my-trace/4665896876_194_cpu_0.1038",
                                                    "workload/my-trace/4665896876_227_cpu_0.1038",
                                                    "workload/my-trace/6162908962_0_cpu_0.1562"
                                                    };
    private static final String[] TRACE_FILE_MEM = {"workload/my-trace/4665896876_28_mem_0.07642", 
                                                    "workload/my-trace/4665896876_43_mem_0.07642",
                                                    "workload/my-trace/4665896876_95_mem_0.07642",
                                                    "workload/my-trace/4665896876_107_mem_0.07642",
                                                    "workload/my-trace/4665896876_173_mem_0.07642",
                                                    "workload/my-trace/4665896876_183_mem_0.07642",
                                                    "workload/my-trace/4665896876_194_mem_0.07642",
                                                    "workload/my-trace/4665896876_227_mem_0.07642",
                                                    "workload/my-trace/6162908962_0_mem_0.1399"
                                                    };

    private static final String[] TRACE_ANOMALY_CPU = {"workload/my-trace/anomaly/3727144341_2_cpu_0.125",
                                                       "workload/my-trace/anomaly/3727144341_8_cpu_0.125",
                                                       "workload/my-trace/anomaly/3769734721_0_cpu_0.1943",
                                                       "workload/my-trace/anomaly/3769734721_2_cpu_0.1943"
                                                      };

    private static final String[] TRACE_ANOMALY_MEM = {"workload/my-trace/anomaly/3727144341_2_mem_0.004662",
                                                       "workload/my-trace/anomaly/3727144341_8_mem_0.004662",
                                                       "workload/my-trace/anomaly/3769734721_0_mem_0.1375",
                                                       "workload/my-trace/anomaly/3769734721_2_mem_0.1375"
                                                      };

    Random getRand = new Random(400);
    
    boolean anomaly = true;

    private static final String sampleTracePth = workdir + "resources/workload/sample/sample.csv";

    int lastTime = 0;

    private int createsVms;

    ArrayList<Double> recordCpu = new ArrayList<Double>();
    ArrayList<Double> recordRam = new ArrayList<Double>();
    
    AnomalyModel createAnomaly =  new AnomalyModel(TRACE_ANOMALY_CPU, TRACE_ANOMALY_MEM, getRand, SCHEDULING_INTERVAL, VM_PES);
    
    SlaStatistic sla;

    public static void main(String[] args) {
        new VerticalVmCpuScalingExampleRevised();
    }

    /**
     * Default constructor that builds the simulation scenario and starts the simulation.
     */
    private VerticalVmCpuScalingExampleRevised() {
        /*Enables just some level of log messages.
          Make sure to import org.cloudsimplus.util.Log;*/
        //Log.setLevel(ch.qos.logback.classic.Level.WARN);

        hostList = new ArrayList<>(HOSTS);
        vmList = new ArrayList<>(VMS);
        cloudletList = new ArrayList<>(CLOUDLETS);

        simulation = new CloudSim();
        simulation.addOnClockTickListener(this::onClockTickListener);

        createDatacenter();
        broker0 = new DatacenterBrokerSimple(simulation);

        vmList.addAll(createListOfScalableVms(VMS));

        sla = new SlaStatistic(vmList);

        createCloudletListsWithDifferentDelays();
        broker0.submitVmList(vmList);
        broker0.submitCloudletList(cloudletList);

        simulation.start();

        printSimulationResults();
    }

    /**
     * Shows updates every time the simulation clock advances.
     * @param evt information about the event happened (that for this Listener is just the simulation time)
     */
    private void onClockTickListener(EventInfo evt) {
        /*vmList.forEach(vm ->
            System.out.printf(
                "\t\tTime %6.1f: Vm %d CPU Usage: %6.2f%% (%2d vCPUs. Running Cloudlets: #%d). Ram Usage: %6.2f%% (%4d of %4d MB)" + " | Host Ram Allocation: %6.2f%% (%5d of %5d MB).%n",
                evt.getTime(), vm.getId(), vm.getCpuPercentUtilization()*100.0, vm.getNumberOfPes(),
                vm.getCloudletScheduler().getCloudletExecList().size(),
                vm.getRam().getPercentUtilization()*100, vm.getRam().getAllocatedResource(), vm.getRam().getCapacity(),
                vm.getHost().getRam().getPercentUtilization() * 100,
                vm.getHost().getRam().getAllocatedResource(),
                vm.getHost().getRam().getCapacity())
        );*/
        /*System.out.println("allocatedResource " + vmList.get(0).getPeVerticalScaling().getAllocatedResource());
        System.out.println("PES x util = " + vmList.get(0).getNumberOfPes()*vmList.get(0).getCpuPercentUtilization());*/
        //统计sla违反率
        
        sla.getViloate(evt);

        int nowTime = (int) evt.getTime();
        recordCpu.add(vmList.get(0).getCpuPercentUtilization());
        recordRam.add(vmList.get(0).getRam().getPercentUtilization());
        int sample_interval = 1;
        if(nowTime % sample_interval == 0 && nowTime != lastTime && nowTime < sample_interval * 9000){
            lastTime = nowTime;
            String time = "" + nowTime + ",";
            String cpuUsage = "" + recordCpu.stream().mapToDouble(a -> a).average().getAsDouble() + ",";
            String ramUsage = "" + recordRam.stream().mapToDouble(a -> a).average().getAsDouble() + "\n";
            File file = new File(sampleTracePth);
            try {
                OutputStream output = new FileOutputStream(file, true);
                output.write((time + cpuUsage + ramUsage).getBytes());
                output.close();
            } catch (Exception e) {
                System.out.println("Exception thrown  :" + e);
            }
            recordCpu = new ArrayList<Double>();
            recordRam = new ArrayList<Double>();
        }
    }

    private void printSimulationResults() {
        final List<Cloudlet> finishedCloudlets = broker0.getCloudletFinishedList();
        final Comparator<Cloudlet> sortByVmId = comparingDouble(c -> c.getVm().getId());
        final Comparator<Cloudlet> sortByStartTime = comparingDouble(Cloudlet::getExecStartTime);
        finishedCloudlets.sort(sortByVmId.thenComparing(sortByStartTime));

        new CloudletsTableBuilder(finishedCloudlets).build();
    }

    /**
     * Creates a Datacenter and its Hosts.
     */
    private void createDatacenter() {
        for (int i = 0; i < HOSTS; i++) {
            hostList.add(createHost());
        }

        Datacenter dc0 = new DatacenterSimple(simulation, hostList, new VmAllocationPolicySimple());
        dc0.setSchedulingInterval(SCHEDULING_INTERVAL);
    }

    private Host createHost() {
        List<Pe> peList = new ArrayList<>(HOST_PES);
        for (int i = 0; i < HOST_PES; i++) {
            peList.add(new PeSimple(1000, new PeProvisionerSimple()));
        }

        final long ram = 20000; //in Megabytes
        final long bw = 100000; //in Megabytes
        final long storage = 10000000; //in Megabites/s
        final int id = hostList.size();
        return new HostSimple(ram, bw, storage, peList)
            .setRamProvisioner(new ResourceProvisionerSimple())
            .setBwProvisioner(new ResourceProvisionerSimple())
            .setVmScheduler(new VmSchedulerTimeShared());
    }

    /**
     * Creates a list of initial VMs in which each VM is able to scale vertically
     * when it is over or underloaded.
     *
     * @param numberOfVms number of VMs to create
     * @return the list of scalable VMs
     * @see #createVerticalPeScaling()
     */
    private List<Vm> createListOfScalableVms(final int numberOfVms) {
        List<Vm> newList = new ArrayList<>(numberOfVms);
        for (int i = 0; i < numberOfVms; i++) {
            Vm vm = createVm();
            vm.setPeVerticalScaling(createVerticalPeScaling());
            //.setRamVerticalScaling(createVerticalRamScaling());
            newList.add(vm);
        }
        return newList;
    }

    /**
     * Creates a Vm object.
     *
     * @return the created Vm
     */
    private Vm createVm() {
        final int id = createsVms++;

        return new VmSimple(id, 1000, VM_PES)
            .setRam(VM_RAM).setBw(1000).setSize(10000)
            .setCloudletScheduler(new CloudletSchedulerTimeShared());
    }

    /**
     * Creates a {@link VerticalVmScaling} for scaling VM's CPU when it's under or overloaded.
     *
     * <p>Realize the lower and upper thresholds are defined inside this method by using
     * references to the methods {@link #lowerCpuUtilizationThreshold(Vm)}
     * and {@link #upperCpuUtilizationThreshold(Vm)}.
     * These methods enable defining thresholds in a dynamic way
     * and even different thresholds for distinct VMs.
     * Therefore, it's a powerful mechanism.
     * </p>
     *
     * <p>
     * However, if you are defining thresholds in a static way,
     * and they are the same for all VMs, you can use a Lambda Expression
     * like below, for instance, instead of creating a new method that just returns a constant value:<br>
     * {@code verticalCpuScaling.setLowerThresholdFunction(vm -> 0.4);}
     * </p>
     *
     * @see #createListOfScalableVms(int)
     */
    private VerticalVmScaling createVerticalPeScaling() {
        //The percentage in which the number of PEs has to be scaled
        final double scalingFactor = 0.1;
        VerticalVmScalingSimple verticalCpuScaling = new VerticalVmScalingSimple(Processor.class, scalingFactor);

        /* By uncommenting the line below, you will see that, instead of gradually
         * increasing or decreasing the number of PEs, when the scaling object detects
         * the CPU usage is above or below the defined thresholds,
         * it will automatically calculate the number of PEs to add/remove to
         * move the VM from the over or underload condition.
        */
        //verticalCpuScaling.setResourceScaling(new ResourceScalingInstantaneous());

        /** Different from the commented line above, the line below implements a ResourceScaling using a Lambda Expression.
         * It is just an example which scales the resource twice the amount defined by the scaling factor
         * defined in the constructor.
         *
         * Realize that if the setResourceScaling method is not called, a ResourceScalingGradual will be used,
         * which scales the resource according to the scaling factor.
         * The lower and upper thresholds after this line can also be defined using a Lambda Expression.
         *
         * So, here we are defining our own {@link ResourceScaling} instead of
         * using the available ones such as the {@link ResourceScalingGradual}
         * or {@link ResourceScalingInstantaneous}.
         */
        //verticalCpuScaling.setResourceScaling(vs -> 2*vs.getScalingFactor()*vs.getAllocatedResource());
        verticalCpuScaling.setResourceScaling(vs -> 1);
        //Allocated = vCPUs x cpu usage
        verticalCpuScaling.setLowerThresholdFunction(this::lowerCpuUtilizationThreshold);
        verticalCpuScaling.setUpperThresholdFunction(this::upperCpuUtilizationThreshold);

        return verticalCpuScaling;
    }

    private VerticalVmScaling createVerticalRamScaling() {
        double scalingFactor = 0.1;
        VerticalVmScalingSimple verticalRamScaling = new VerticalVmScalingSimple(Ram.class, scalingFactor);
        /* By uncommenting the line below, you will see that, instead of gradually
         * increasing or decreasing the RAM, when the scaling object detects
         * the RAM usage is above or below the defined thresholds,
         * it will automatically calculate the amount of RAM to add/remove to
         * move the VM from the over or underload condition.
        */
        //verticalRamScaling.setResourceScaling(new ResourceScalingInstantaneous());
        verticalRamScaling.setResourceScaling(vs -> 1);
        verticalRamScaling.setLowerThresholdFunction(this::lowerRamUtilizationThreshold);
        verticalRamScaling.setUpperThresholdFunction(this::upperRamUtilizationThreshold);
        return verticalRamScaling;
    }

    /**
     * Defines the minimum CPU utilization percentage that indicates a Vm is underloaded.
     * This function is using a statically defined threshold, but it would be defined
     * a dynamic threshold based on any condition you want.
     * A reference to this method is assigned to each Vertical VM Scaling created.
     *
     * @param vm the VM to check if its CPU is underloaded.
     *        <b>The parameter is not being used internally, which means the same
     *        threshold is used for any Vm.</b>
     * @return the lower CPU utilization threshold
     * @see #createVerticalPeScaling()
     */
    private double lowerCpuUtilizationThreshold(Vm vm) {
        return 0.4;
    }

    /**
     * Defines the maximum CPU utilization percentage that indicates a Vm is overloaded.
     * This function is using a statically defined threshold, but it would be defined
     * a dynamic threshold based on any condition you want.
     * A reference to this method is assigned to each Vertical VM Scaling created.
     *
     * @param vm the VM to check if its CPU is overloaded.
     *        The parameter is not being used internally, that means the same
     *        threshold is used for any Vm.
     * @return the upper CPU utilization threshold
     * @see #createVerticalPeScaling()
     */
    private double upperCpuUtilizationThreshold(Vm vm) {
        return 0.8;
    }

    private double lowerRamUtilizationThreshold(Vm vm) {
        return 0.5;
    }

    private double upperRamUtilizationThreshold(Vm vm) {
        return 0.7;
    }

    /**
     * Creates lists of Cloudlets to be submitted to the broker with different delays,
     * simulating their arrivals at different times.
     * Adds all created Cloudlets to the {@link #cloudletList}.
     */
    private void createCloudletListsWithDifferentDelays() {
        final int normalCloudletsNumber = (int)(CLOUDLETS*9.0/10);
        final int anomalyCloudletsNumber = CLOUDLETS-normalCloudletsNumber;

        int delayTimeAbs = 0;

        //Creates a List of Cloudlets that will start running immediately when the simulation starts
        for (int i = 0; i < normalCloudletsNumber; i++) {
            double nextDouble = getRand.nextDouble(1) * (
                CLOUDLETS_LENGTH_UPPER - CLOUDLETS_LENGTH_LOWER) + CLOUDLETS_LENGTH_LOWER;
            delayTimeAbs = delayTimeAbs + nextTime(1.0/4);
            cloudletList.add(createCloudlet(CLOUDLETS_INITIAL_LENGTH+((int) nextDouble), 1, delayTimeAbs));
        }

        //注入异常
        if (anomaly == true) {
        delayTimeAbs = 300;
        for (int i = 1; i <= anomalyCloudletsNumber; i++) {
            double nextDouble = getRand.nextDouble(1) * (
                CLOUDLETS_LENGTH_UPPER - CLOUDLETS_LENGTH_LOWER) + CLOUDLETS_LENGTH_LOWER;
            delayTimeAbs = delayTimeAbs + nextTime(1.0/300);
            String type = "bottleneck"; //patternShift只支持mixed
            String resource = "mix";
            cloudletList.add(createAnomaly.generateAnomaly((int) nextDouble, 1, resource, type, delayTimeAbs));
        }
        }
    }

    /**
     * Creates a single Cloudlet with no delay, which means the Cloudlet arrival time will
     * be zero (exactly when the simulation starts).
     *
     * @param length the Cloudlet length
     * @param numberOfPes the number of PEs the Cloudlets requires
     * @return the created Cloudlet
     */

    /**
     * Creates a single Cloudlet.
     *
     * @param length the length of the cloudlet to create.
     * @param numberOfPes the number of PEs the Cloudlets requires.
     * @param delay the delay that defines the arrival time of the Cloudlet at the Cloud infrastructure.
     * @return the created Cloudlet
     */
    private Cloudlet createCloudlet(final long length, final int numberOfPes, final double delay) {
        /*
        Since a VM PE isn't used by two Cloudlets at the same time,
        the Cloudlet can used 100% of that CPU capacity at the time
        it is running. Even if a CloudletSchedulerTimeShared is used
        to share the same VM PE among multiple Cloudlets,
        just one Cloudlet uses the PE at a time.
        Then it is preempted to enable other Cloudlets to use such a VM PE.
         */
        
        int number = getRand.nextInt(9); 
        final UtilizationModel utilizationCpu = UtilizationModelPlanetLab.getInstance(TRACE_FILE_CPU[number], SCHEDULING_INTERVAL);
        final UtilizationModel utilizationMem = UtilizationModelPlanetLab.getInstance(TRACE_FILE_MEM[number], SCHEDULING_INTERVAL, Unit.ABSOLUTE);
        
        /**
         * Since BW e RAM are shared resources that don't enable preemption,
         * two Cloudlets can't use the same portion of such resources at the same time
         * (unless virtual memory is enabled, but such a feature is not available in simulation).
         * This way, the total capacity of such resources is being evenly split among created Cloudlets.
         * If there are 10 Cloudlets, each one will use just 10% of such resources.
         * This value can be defined in different ways, as you want. For instance, some Cloudlets
         * can require more resources than other ones.
         * To enable that, you would need to instantiate specific {@link UtilizationModelDynamic} for each Cloudlet,
         * use a {@link UtilizationModelStochastic} to define resource usage randomly,
         * or use any other {@link UtilizationModel} implementation.
        */
        final UtilizationModel utilizationModelDynamic = new UtilizationModelDynamic(1.0/CLOUDLETS);
        Cloudlet cl = new CloudletSimple(length, numberOfPes);
        cl.setFileSize(1024)
            .setOutputSize(1024)
            .setUtilizationModelBw(utilizationModelDynamic)
            .setUtilizationModelRam(utilizationMem)
            .setUtilizationModelCpu(utilizationCpu)
            .setSubmissionDelay(delay);
            
        return cl;
    }

    private int nextTime(double rateParameter){
    return (int) (-Math.log(1.0 - getRand.nextDouble(1)) / rateParameter);
    }
}

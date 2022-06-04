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
package org.cloudsimplus.autoscaling;

import org.cloudbus.cloudsim.brokers.DatacenterBroker;
import org.cloudbus.cloudsim.cloudlets.Cloudlet;
import org.cloudbus.cloudsim.cloudlets.CloudletExecution;
import org.cloudbus.cloudsim.resources.*;
import org.cloudbus.cloudsim.vms.Vm;
import org.cloudsimplus.autoscaling.resources.ResourceScaling;
import org.cloudsimplus.autoscaling.resources.ResourceScalingGradual;
import org.cloudbus.cloudsim.resources.Ram;

/**
 * A {@link VerticalVmScaling} implementation which allows a {@link DatacenterBroker}
 * to perform on demand up or down scaling for some {@link Vm} resource, such as {@link Ram},
 * {@link Pe} or {@link Bandwidth}.
 *
 * <p>For each resource that is required to be scaled, a distinct {@link VerticalVmScaling}
 * instance must be assigned to the VM to be scaled.</p>
 *
 * @author Manoel Campos da Silva Filho
 * @since CloudSim Plus 1.1.0
 */
public class VerticalVmScalingSimple extends VerticalVmScalingAbstract {

    /**
     * Creates a VerticalVmScalingSimple with a {@link ResourceScalingGradual} scaling type.
     *
     * @param resourceClassToScale the class of Vm resource that this scaling object will request
     *                             up or down scaling (such as {@link Ram}.class,
     *                             {@link Bandwidth}.class or {@link Processor}.class).
     * @param scalingFactor the factor (a percentage value in scale from 0 to 1)
     *                      that will be used to scale a Vm resource up or down,
     *                      whether such a resource is over or underloaded, according to the
     *                      defined predicates.
     *                      In the case of up scaling, the value 1 will scale the resource in 100%,
     *                      doubling its capacity.
     * @see VerticalVmScaling#setResourceScaling(ResourceScaling)
     */
    double lastMipsUsage = 1.;
    double lastPeUsage= 1.;
    int lastTime = 0;
    public VerticalVmScalingSimple(final Class<? extends ResourceManageable> resourceClassToScale, final double scalingFactor){
        super(resourceClassToScale, new ResourceScalingGradual(), scalingFactor);
    }

    @Override
    public boolean isVmUnderloaded() {
        if (getResourceClass().getSimpleName().equals("Processor")){
            double cpuPercentage = getVm().getCpuPercentUtilization();
            int runningCloudlet = getVm().getCloudletScheduler().getCloudletExecList().size();
            if (lastTime != (int) getVm().getSimulation().clock()){
            /*System.out.printf(
                    "\t\tTime %6.1f: Vm %d CPU Usage: %6.2f%% (%2d vCPUs. Running Cloudlets: #%d). RAM usage: %.2f%% (%d MB)%n",
                    getVm().getSimulation().clock(), getVm().getId(), getVm().getCpuPercentUtilization()*100.0, getVm().getNumberOfPes(),
                    getVm().getCloudletScheduler().getCloudletExecList().size(),
                    getVm().getRam().getPercentUtilization()*100, getVm().getRam().getAllocatedResource());*/
                    lastTime = (int) getVm().getSimulation().clock();
            }
            double latestPercentage = lastMipsUsage * 1 / lastPeUsage * getPePercentage();
            if (cpuPercentage != 0. && runningCloudlet != 0 && getPePercentage() != 0.){
                lastMipsUsage = cpuPercentage;
                lastPeUsage = getPePercentage();
            }
            double AmountPes = getVm().getPeVerticalScaling().getResourceAmountToScale();
            if (getResource().getCapacity() -  AmountPes > 0){
                return latestPercentage < getLowerThresholdFunction().apply(getVm());
            }
            else{
                return false;
            }
        
        //按照pe个数占用的扩展方案
        /*double AmountPes = getVm().getPeVerticalScaling().getResourceAmountToScale();
        long cloudletRequestMinPes = 100;
        for (CloudletExecution cloud : getVm().getCloudletScheduler().getCloudletExecList()) {
            long pes = cloud.getNumberOfPes();
            if (pes < cloudletRequestMinPes){
                cloudletRequestMinPes = pes;
            }
        }
        
        if (getResource().getCapacity() -  AmountPes > 0){
            return getResource().getPercentUtilization() < getLowerThresholdFunction().apply(getVm());
        }
        else{
            return false;
        }*/
        }else{
            //return getVm().getRam().getPercentUtilization() < getLowerThresholdFunction().apply(getVm());
            return getResource().getPercentUtilization() < getLowerThresholdFunction().apply(getVm());
            
        }
    }

    @Override
    public boolean isVmOverloaded() {
        if (getResourceClass().getSimpleName().equals("Processor")){
            double cpuPercentage = getVm().getCpuPercentUtilization();
            int runningCloudlet = getVm().getCloudletScheduler().getCloudletExecList().size();
            
            double latestPercentage = lastMipsUsage * 1 / lastPeUsage * getPePercentage();
            if (cpuPercentage != 0. && runningCloudlet != 0 && getPePercentage() != 0.){
                lastMipsUsage = cpuPercentage;
                lastPeUsage = getPePercentage();
            }
            
            return latestPercentage > getUpperThresholdFunction().apply(getVm());
        }else{
            return getResource().getPercentUtilization() > getUpperThresholdFunction().apply(getVm());
            //return getVm().getRam().getPercentUtilization() > getUpperThresholdFunction().apply(getVm());
        }
    }

    public double getPePercentage() {
        long capacity = getVm().getNumberOfPes();
        double allocate = 0.;
        for (CloudletExecution cloud : getVm().getCloudletScheduler().getCloudletExecList()) {
            allocate += cloud.getNumberOfPes();
        }
        double PePercentage = allocate / capacity;
        if(PePercentage > 1){
            PePercentage = 1.;
        }
        return PePercentage;
    }
}

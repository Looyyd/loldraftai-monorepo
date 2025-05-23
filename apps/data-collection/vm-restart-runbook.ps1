# Runbook used in azure to automatically restart a spot VM if it is deallocated.

Param(
    [Parameter(Mandatory = $true)]
    [string]$SubscriptionId,

    [Parameter(Mandatory = $true)]
    [string]$VmName,
    
    [Parameter(Mandatory = $true)]
    [string]$ResourceGroupName
)

# Connect using managed identity
try {
    Disable-AzContextAutosave -Scope Process
    Connect-AzAccount -Identity
    Set-AzContext -Subscription $SubscriptionId
    
    Write-Output "Successfully connected using Managed Identity"
} catch {
    Write-Error "Failed to connect: $_"
    throw
}

# Get VM status
$vm = Get-AzVM -ResourceGroupName $ResourceGroupName -Name $VmName -Status
$powerState = ($vm.Statuses | Where-Object { $_.Code -match 'PowerState' }).DisplayStatus

Write-Output "Current VM state: $powerState"

# If VM is deallocated, try to start it
if ($powerState -eq "VM deallocated") {
    Write-Output "Attempting to start VM $VmName"
    Start-AzVM -ResourceGroupName $ResourceGroupName -Name $VmName
}

# Get final state
$vm = Get-AzVM -ResourceGroupName $ResourceGroupName -Name $VmName -Status
$finalState = ($vm.Statuses | Where-Object { $_.Code -match 'PowerState' }).DisplayStatus
Write-Output "Final VM state: $finalState"

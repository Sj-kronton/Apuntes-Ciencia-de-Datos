from azure.iot.device import ProvisioningDeviceClient

ID_SCOPE = "0ne010F6F29"
DEVICE_ID = "290lbx1pawb"
SYMMETRIC_KEY = "Yz7MOHLHMV4v7iNimoJCmD8GSfMkiAww5ktRK3sOMTM="
MODEL_ID = "dtmi:granjadecacao:AmbienteDeDoselYCampo_rn;1"

provisioning_client = ProvisioningDeviceClient.create_from_symmetric_key(
    provisioning_host="global.azure-devices-provisioning.net",
    registration_id=DEVICE_ID,
    id_scope=ID_SCOPE,
    symmetric_key=SYMMETRIC_KEY,
    websockets=True,
)
provisioning_client.provisioning_payload = {"modelId": MODEL_ID}
registration_result = provisioning_client.register()

print("Estado:", registration_result.status)
print("Hub asignado:", registration_result.registration_state.assigned_hub)

#iotc-d56f0dee-f8d2-4958-9948-b3d323c4f2d9.azure-devices.net
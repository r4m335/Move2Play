using UnityEngine;
using UnityEngine.UI;

public class WebcamTest : MonoBehaviour
{
    public RawImage webcamImage;

    private WebCamTexture webcam;

    void Start()
    {
        Debug.Log("Devices: " + WebCamTexture.devices.Length);

        foreach (var d in WebCamTexture.devices)
        {
            Debug.Log("Camera: " + d.name);
        }

        webcam = new WebCamTexture(
            WebCamTexture.devices[0].name,
            1280,
            720,
            30
        );

        webcam.Play();

        webcamImage.texture = webcam;

        Debug.Log("Started webcam");
    }

    void Update()
    {
        Debug.Log(
            $"Width={webcam.width} Height={webcam.height} Updated={webcam.didUpdateThisFrame}"
        );
    }
}
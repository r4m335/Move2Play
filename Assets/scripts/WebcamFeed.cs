using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Captures webcam feed and displays it on a RawImage.
/// Attach this to any GameObject, then drag the target RawImage into the inspector.
/// </summary>
public class WebcamFeed : MonoBehaviour
{
    [Tooltip("RawImage component where the webcam feed will be displayed")]
    public RawImage targetRawImage;

    [Tooltip("Desired width (will keep aspect ratio)")]
    public int width = 640;

    [Tooltip("Desired height (will keep aspect ratio)")]
    public int height = 480;

    [Tooltip("Frames per second for the webcam")]
    public int fps = 30;

    private WebCamTexture webcamTexture;
    private WebCamDevice[] devices;

    void Start()
    {
        devices = WebCamTexture.devices;
        if (devices.Length == 0)
        {
            Debug.LogError("WebcamFeed: No webcam detected!");
            enabled = false;
            return;
        }

        // Use the first available camera (you can modify to select a specific one)
        string cameraName = devices[0].name;
        webcamTexture = new WebCamTexture(cameraName, width, height, fps);

        if (targetRawImage != null)
        {
            targetRawImage.texture = webcamTexture;
            targetRawImage.material.mainTexture = webcamTexture;
        }

        webcamTexture.Play();
        Debug.Log($"WebcamFeed started: {cameraName} ({width}x{height} @ {fps}fps)");
    }

    void Update()
    {
        // Optionally flip the image horizontally if needed (mirror effect)
        if (targetRawImage != null && webcamTexture != null && webcamTexture.didUpdateThisFrame)
        {
            // Uncomment if you want to mirror the webcam image
            // targetRawImage.uvRect = new Rect(1, 0, -1, 1);
        }
    }

    void OnDestroy()
    {
        if (webcamTexture != null && webcamTexture.isPlaying)
        {
            webcamTexture.Stop();
            webcamTexture = null;
        }
    }
}
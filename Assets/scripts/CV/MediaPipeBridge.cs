using System.Collections.Generic;
using Mediapipe.Tasks.Components.Containers;
using UnityEngine;

public class MediaPipeBridge : MonoBehaviour
{
    public static MediaPipeBridge Instance;

    private readonly List<NormalizedLandmark> currentLandmarks = new List<NormalizedLandmark>();
    private readonly object landmarkLock = new object();

    void Awake()
    {
        if (Instance == null)
            Instance = this;
        else
            Destroy(gameObject);
    }

    public void SetLandmarks(IList<NormalizedLandmark> landmarks)
    {
        lock (landmarkLock)
        {
            currentLandmarks.Clear();
            if (landmarks != null)
            {
                foreach (var lm in landmarks)
                {
                    currentLandmarks.Add(lm);
                }
            }
        }
    }

    public List<NormalizedLandmark> GetLandmarks()
    {
        lock (landmarkLock)
        {
            return new List<NormalizedLandmark>(currentLandmarks);
        }
    }
}
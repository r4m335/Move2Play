using UnityEngine;
using System.Collections.Generic;

public class MediaPipePoseManager : MonoBehaviour
{
    public static MediaPipePoseManager Instance { get; private set; }

    private List<PoseLandmark> currentLandmarks = new List<PoseLandmark>();

    void Awake()
    {
        if (Instance == null)
            Instance = this;
        else
            Destroy(gameObject);
    }

    public void UpdateLandmarks(List<PoseLandmark> landmarks)
    {
        if (landmarks != null && landmarks.Count >= 33)
            currentLandmarks = new List<PoseLandmark>(landmarks);
        else
            currentLandmarks = new List<PoseLandmark>();
    }

    public List<PoseLandmark> GetCurrentLandmarks()
    {
        return new List<PoseLandmark>(currentLandmarks);
    }
}
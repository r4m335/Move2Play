using UnityEngine;

[System.Serializable]
public struct PoseLandmark
{
    public Vector3 position;
    public float visibility;

    public PoseLandmark(Vector3 pos, float vis)
    {
        position = pos;
        visibility = vis;
    }
}
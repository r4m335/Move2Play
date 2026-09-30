using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class PoseVisualizer : MonoBehaviour
{
    [Header("References")]
    public MediaPipePoseManager poseManager;  // Drag the PoseManager object here
    public RectTransform canvasRect;          // Your Canvas (for coordinate conversion)

    [Header("Appearance")]
    public GameObject jointPrefab;            // A UI Image prefab (circle)
    public GameObject linePrefab;             // A UI Image prefab (thin rectangle, or use LineRenderer)
    public Color jointColor = Color.green;
    public Color lineColor = Color.white;
    public float jointSize = 10f;
    public float lineWidth = 2f;

    [Header("Performance")]
    public float updateInterval = 0.033f;     // ~30 FPS

    private List<RectTransform> joints = new List<RectTransform>();
    private List<Image> lines = new List<Image>();
    private float lastUpdateTime;
    private bool isInitialized = false;

    // MediaPipe skeleton connections (indices)
    private readonly (int from, int to)[] connections = new (int, int)[]
    {
        // Torso
        (11,12), (12,24), (11,23), (23,24),
        // Arms
        (11,13), (13,15), (12,14), (14,16),
        // Legs
        (23,25), (25,27), (24,26), (26,28),
        // Face (optional)
        (0,1), (0,2), (1,3), (2,4)
    };

    void Start()
    {
        if (poseManager == null)
            poseManager = FindObjectOfType<MediaPipePoseManager>();
        if (canvasRect == null)
            canvasRect = GetComponentInParent<Canvas>().GetComponent<RectTransform>();

        if (jointPrefab == null)
            CreateDefaultPrefabs();

        CreateVisuals();
        isInitialized = true;
    }

    void Update()
    {
        if (!isInitialized) return;
        if (Time.time - lastUpdateTime < updateInterval) return;
        lastUpdateTime = Time.time;

        var landmarks = poseManager?.GetCurrentLandmarks();
        if (landmarks == null || landmarks.Count < 33)
        {
            HideVisuals();
            return;
        }

        UpdateJointPositions(landmarks);
        ShowVisuals();
    }

    void CreateVisuals()
    {
        // Create joints
        for (int i = 0; i < 33; i++)
        {
            GameObject jointObj = Instantiate(jointPrefab, transform);
            RectTransform rt = jointObj.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(jointSize, jointSize);
            Image img = jointObj.GetComponent<Image>();
            img.color = jointColor;
            joints.Add(rt);
        }

        // Create lines
        foreach (var conn in connections)
        {
            GameObject lineObj = Instantiate(linePrefab, transform);
            Image lineImg = lineObj.GetComponent<Image>();
            lineImg.color = lineColor;
            lines.Add(lineImg);
        }
    }

    void UpdateJointPositions(List<PoseLandmark> landmarks)
    {
        for (int i = 0; i < 33; i++)
        {
            if (i >= joints.Count) break;

            Vector2 screenPos = WorldToCanvasPoint(landmarks[i].position);
            joints[i].anchoredPosition = screenPos;

            // Optional: alpha based on visibility
            Color col = joints[i].GetComponent<Image>().color;
            col.a = landmarks[i].visibility > 0.5f ? 1f : 0.3f;
            joints[i].GetComponent<Image>().color = col;
        }

        // Update lines
        for (int idx = 0; idx < connections.Length; idx++)
        {
            if (idx >= lines.Count) break;
            var conn = connections[idx];
            if (conn.from >= joints.Count || conn.to >= joints.Count) continue;

            Vector2 fromPos = joints[conn.from].anchoredPosition;
            Vector2 toPos = joints[conn.to].anchoredPosition;
            DrawLine(lines[idx], fromPos, toPos, lineWidth);
        }
    }

    void DrawLine(Image lineImg, Vector2 from, Vector2 to, float thickness)
    {
        Vector2 dir = (to - from).normalized;
        float distance = Vector2.Distance(from, to);

        RectTransform rt = lineImg.rectTransform;
        rt.anchoredPosition = from + dir * distance * 0.5f;
        rt.sizeDelta = new Vector2(distance, thickness);
        rt.localRotation = Quaternion.FromToRotation(Vector2.right, dir);
    }

    Vector2 WorldToCanvasPoint(Vector3 worldPos)
    {
        // worldPos is normalized MediaPipe coordinate (0..1, Y up)
        // Convert to canvas anchored position (0..width, 0..height)
        float canvasWidth = canvasRect.rect.width;
        float canvasHeight = canvasRect.rect.height;
        return new Vector2(worldPos.x * canvasWidth, (1f - worldPos.y) * canvasHeight);
    }

    void CreateDefaultPrefabs()
    {
        // Create joint prefab (circle)
        jointPrefab = new GameObject("JointPrefab");
        jointPrefab.AddComponent<CanvasRenderer>();
        Image img = jointPrefab.AddComponent<Image>();
        img.sprite = CreateCircleSprite();
        img.color = jointColor;
        jointPrefab.transform.SetParent(transform, false);
        jointPrefab.SetActive(false);

        // Create line prefab (thin rectangle)
        linePrefab = new GameObject("LinePrefab");
        linePrefab.AddComponent<CanvasRenderer>();
        Image lineImg = linePrefab.AddComponent<Image>();
        lineImg.color = lineColor;
        linePrefab.transform.SetParent(transform, false);
        linePrefab.SetActive(false);
    }

    Sprite CreateCircleSprite()
    {
        Texture2D tex = new Texture2D(4, 4);
        for (int x = 0; x < 4; x++)
            for (int y = 0; y < 4; y++)
                tex.SetPixel(x, y, Color.white);
        tex.Apply();
        return Sprite.Create(tex, new Rect(0, 0, 4, 4), new Vector2(0.5f, 0.5f));
    }

    void ShowVisuals()
    {
        foreach (var j in joints) j.gameObject.SetActive(true);
        foreach (var l in lines) l.gameObject.SetActive(true);
    }

    void HideVisuals()
    {
        foreach (var j in joints) j.gameObject.SetActive(false);
        foreach (var l in lines) l.gameObject.SetActive(false);
    }
}
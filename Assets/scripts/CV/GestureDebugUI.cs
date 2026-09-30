using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;
using System.Linq;

public class GestureDebugUI : MonoBehaviour
{
    [Header("References")]
    public RuleBasedGestureRecognizer recognizer;  // set from PlayerController or CalibrationManager
    public Transform contentParent;                // parent for the gesture rows
    public GameObject gestureRowPrefab;            // a prefab with: TMP text for name, Slider for confidence

    [Header("Additional Stats")]
    public TMP_Text motionText;
    public TMP_Text jumpImpulseText;
    public TMP_Text abductionText;
    public TMP_Text asymmetryText;

    private Dictionary<string, GestureRow> gestureRows = new Dictionary<string, GestureRow>();
    private float lastUpdateTime = 0f;
    private float updateInterval = 0.1f; // update UI 10 times per second

    // This will be filled by the recognizer's output
    public class GestureRow
    {
        public TMP_Text nameText;
        public Slider confidenceSlider;
        public TMP_Text percentText;
    }

    void Start()
    {
        // Pre-create rows for all known gestures (from GestureActionMap keys)
        var gestures = new List<string> { "idle", "run", "punch", "kick", "block", "lean_left", "lean_right", "squat", "jump" };
        foreach (var g in gestures)
        {
            var row = Instantiate(gestureRowPrefab, contentParent).GetComponent<GestureRow>();
            if (row == null)
                Debug.LogError("Gesture row prefab must have a GestureRow component");

            row.nameText.text = g.ToUpper();
            gestureRows[g] = row;
        }
    }

    void Update()
    {
        if (recognizer == null) return;
        if (Time.time - lastUpdateTime < updateInterval) return;
        lastUpdateTime = Time.time;

        // Get the latest frame result from the recognizer's internal state
        // For simplicity, we assume you store the last result in a public field.
        // Alternatively, add a public property to RuleBasedGestureRecognizer that exposes
        // lastGesture, lastConfidence, lastRawProbabilities, lastMotionData.
        var lastResult = recognizer.GetLastResult(); // we'll add this method

        if (lastResult == null) return;

        // Update gesture confidence bars
        if (lastResult.allProbabilities != null)
        {
            int idx = 0;
            foreach (var gesture in gestureRows.Keys)
            {
                if (idx < lastResult.allProbabilities.Length)
                {
                    float conf = lastResult.allProbabilities[idx];
                    var row = gestureRows[gesture];
                    row.confidenceSlider.value = conf;
                    row.percentText.text = $"{conf * 100f:F1}%";
                    // Highlight if this is the current gesture
                    if (gesture == lastResult.gesture)
                    {
                        row.nameText.color = Color.green;
                        row.confidenceSlider.fillRect.GetComponent<Image>().color = Color.green;
                    }
                    else
                    {
                        row.nameText.color = Color.white;
                        row.confidenceSlider.fillRect.GetComponent<Image>().color = new Color(0.2f, 0.6f, 1f);
                    }
                }
                idx++;
            }
        }

        // Motion diagnostics
        motionText.text = $"LEG MOTION: {lastResult.legMotion:F4}";
        jumpImpulseText.text = $"JUMP IMPULSE: {lastResult.jumpImpulse:F4}";
        abductionText.text = $"L.ABD: {lastResult.leftAbduction:F3}   R.ABD: {lastResult.rightAbduction:F3}";
        asymmetryText.text = $"LEG ASYM: {lastResult.legAsymmetry:F4}";
    }
}

// A component attached to each row prefab
public class GestureRow : MonoBehaviour
{
    public TMP_Text nameText;
    public Slider confidenceSlider;
    public TMP_Text percentText;
}
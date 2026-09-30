using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using TMPro;
using System.Collections.Generic;
using System.Collections;
using Mediapipe.Tasks.Components.Containers;
using Mediapipe.Unity.Sample.PoseLandmarkDetection;

public class CalibrationManager : MonoBehaviour
{
    public static CalibrationManager Instance;

    [Header("UI – Text & Button")]
    public TMP_Text hintText;
    // gestureText removed
    public TMP_Text confidenceText;
    public TMP_Text statusText;
    public Button startGameButton;
    public Button controlsButton;

    [Header("Progress UI")]
    public TMP_Text progressText;
    public Image progressBarFill;

    [Header("Detection Status")]
    public Image detectionImage;
    public TMP_Text detectionText;
    public Sprite detectedSprite;
    public Sprite holdingSprite;
    public Sprite notDetectedSprite;

    [Header("Body Tracking UI")]
    public Image headDot;
    public Image handsDot;
    public Image kneesDot;
    public Image feetDot;
    public Sprite greenDotSprite;
    public Sprite redDotSprite;

    [Header("Gesture Illustration")]
    public Image gestureIllustration;
    public Sprite idleSprite;
    public Sprite runSprite;
    public Sprite leanLeftSprite;
    public Sprite leanRightSprite;
    public Sprite jumpSprite;
    public Sprite punchSprite;
    public Sprite kickSprite;

    [Header("Success Splash")]
    public Image successSplashImage;

    [Header("Feedback")]
    public TMP_Text feedbackText;

    [Header("Calibration Settings")]
    public float minVisibilityThreshold = 0.6f;
    public float minGestureConfidence = 0.7f;
    public float holdRequiredDuration = 3f;
    public int repeatRequiredCount = 1;
    public string nextSceneName = "Forest";
    public string controlsSceneName = "Guide";

    private const float KICK_ELEVATION_THRESHOLD = 0.03f;
    private const float KICK_MOTION_THRESHOLD = 0.02f;
    private const float KICK_ASYMMETRY_THRESHOLD = 0.20f;
    private const float JUMP_IMPULSE_MAX_FOR_KICK = 0.20f;

    private List<CalibrationStep> steps;
    private int currentStep = 0;
    private RuleBasedGestureRecognizer recognizer;
    private string currentGesture = "idle";
    private float currentConfidence = 0f;
    private string wrongGestureMessage = "";
    private float wrongGestureTimer = 0f;
    private bool tutorialCompleted = false;

    private Queue<float> hipYHistory = new Queue<float>();
    private const int HIP_HISTORY_LEN = 6;
    private bool hipHistoryInitialized = false;

    private List<List<PoseLandmark>> rawLandmarkBuffer = new List<List<PoseLandmark>>();
    private const int FALLBACK_BUFFER_SIZE = 8;

    private Coroutine splashRoutine;

    void Awake()
    {
        if (Instance == null)
            Instance = this;
        else
        {
            Destroy(gameObject);
            return;
        }
    }

    void Start()
    {
        recognizer = new RuleBasedGestureRecognizer();

        // --- UPDATED steps list with new instruction text for left/right ---
        steps = new List<CalibrationStep>
        {
            new CalibrationStep("idle", "Stand naturally,\narms at sides.", holdRequired: true, requiredCount: 1),
            new CalibrationStep("run", "Raise both arms\nwide (T-pose)", holdRequired: true, requiredCount: 1),

            // TURN LEFT
            new CalibrationStep("lean_left",
                "Raise only your\nleft arm sideways",
                holdRequired: true,
                requiredCount: 1),

            // TURN RIGHT
            new CalibrationStep("lean_right",
                "Raise only your\nright arm sideways",
                holdRequired: true,
                requiredCount: 1),

            new CalibrationStep("jump", "Jump", holdRequired: false, requiredCount: 1),
            new CalibrationStep("punch", "Punch forward", holdRequired: false, requiredCount: 1),
            new CalibrationStep("kick", "Kick or\nraise one knee", holdRequired: false, requiredCount: 1),
        };

        startGameButton.interactable = false;
        startGameButton.onClick.AddListener(() => StopWebcamAndLoadScene());
        startGameButton.GetComponentInChildren<TMP_Text>().text = "Play";

        if (controlsButton != null)
            controlsButton.onClick.AddListener(() => SceneManager.LoadScene(controlsSceneName));

        statusText.text = "Calibration starting...";
        hintText.text = "Stand in the centre of the frame.\nIf unsure, press 'Controls' to see the gestures.";

        // Hide confidence text entirely
        if (confidenceText != null)
            confidenceText.gameObject.SetActive(false);

        // Hide the splash image completely
        if (successSplashImage != null)
        {
            Color c = successSplashImage.color;
            c.a = 0f;
            successSplashImage.color = c;
            successSplashImage.gameObject.SetActive(false);
        }

        UpdateProgressUI();
    }

    void Update()
    {
        IList<NormalizedLandmark> rawLandmarks = MediaPipeBridge.Instance?.GetLandmarks();
        if (rawLandmarks == null || rawLandmarks.Count < 33)
        {
            statusText.text = "Waiting for pose data...";
            SetDetectionUI(notDetectedSprite, "NO POSE", Color.red);
            return;
        }

        List<PoseLandmark> poseLandmarks = new List<PoseLandmark>();
        foreach (var lm in rawLandmarks)
        {
            Vector3 pos = new Vector3(lm.x, lm.y, lm.z);
            float vis = lm.visibility ?? 0f;
            poseLandmarks.Add(new PoseLandmark(pos, vis));
        }

        rawLandmarkBuffer.Add(new List<PoseLandmark>(poseLandmarks));
        while (rawLandmarkBuffer.Count > FALLBACK_BUFFER_SIZE)
            rawLandmarkBuffer.RemoveAt(0);

        float jumpImpulse = ComputeJumpImpulse(poseLandmarks);
        float legMotion = ComputeLegMotion(rawLandmarkBuffer);
        float legAsymmetry = ComputeLegAsymmetry(rawLandmarkBuffer);
        float legElevation = ComputeLegElevation(poseLandmarks);

        GestureResult result = recognizer.ProcessFrame(poseLandmarks, Time.deltaTime);
        if (result != null && !string.IsNullOrEmpty(result.gesture))
        {
            currentGesture = result.gesture;
            currentConfidence = result.confidence;
        }

        EvaluateCalibration(poseLandmarks, jumpImpulse, legMotion, legAsymmetry, legElevation);
        UpdateWrongGestureMessage();

        if (tutorialCompleted)
        {
            bool recognizerJump = (currentGesture == "jump" && currentConfidence >= minGestureConfidence);
            bool fallbackJump = jumpImpulse > 0.035f;
            if (recognizerJump || fallbackJump)
            {
                StopWebcamAndLoadScene();
            }
        }
    }

    private float ComputeJumpImpulse(List<PoseLandmark> landmarks)
    {
        if (landmarks.Count < 25) return 0f;
        float hipY = (landmarks[23].position.y + landmarks[24].position.y) * 0.5f;

        if (!hipHistoryInitialized)
        {
            hipYHistory.Clear();
            for (int i = 0; i < HIP_HISTORY_LEN; i++)
                hipYHistory.Enqueue(hipY);
            hipHistoryInitialized = true;
            return 0f;
        }

        hipYHistory.Enqueue(hipY);
        if (hipYHistory.Count > HIP_HISTORY_LEN) hipYHistory.Dequeue();

        float minY = float.MaxValue, maxY = float.MinValue;
        foreach (float y in hipYHistory)
        {
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }
        return maxY - minY;
    }

    private float ComputeLegMotion(List<List<PoseLandmark>> buffer)
    {
        if (buffer.Count < 2) return 0f;
        int[] legIndices = { 23, 24, 25, 26, 27, 28 };
        float totalMotion = 0f;
        int count = 0;
        for (int i = 1; i < buffer.Count; i++)
        {
            var prev = buffer[i - 1];
            var curr = buffer[i];
            foreach (int idx in legIndices)
            {
                if (idx < prev.Count && idx < curr.Count &&
                    prev[idx].visibility > 0.35f && curr[idx].visibility > 0.35f)
                {
                    totalMotion += Vector3.Distance(prev[idx].position, curr[idx].position);
                    count++;
                }
            }
        }
        return count > 0 ? totalMotion / count : 0f;
    }

    private float ComputeLegAsymmetry(List<List<PoseLandmark>> buffer)
    {
        if (buffer.Count < 2) return 0f;
        float leftMotion = 0f, rightMotion = 0f;
        int leftCount = 0, rightCount = 0;
        int[] leftIndices = { 23, 25, 27 };
        int[] rightIndices = { 24, 26, 28 };
        for (int i = 1; i < buffer.Count; i++)
        {
            var prev = buffer[i - 1];
            var curr = buffer[i];
            foreach (int idx in leftIndices)
            {
                if (idx < prev.Count && idx < curr.Count &&
                    prev[idx].visibility > 0.35f && curr[idx].visibility > 0.35f)
                {
                    leftMotion += Vector3.Distance(prev[idx].position, curr[idx].position);
                    leftCount++;
                }
            }
            foreach (int idx in rightIndices)
            {
                if (idx < prev.Count && idx < curr.Count &&
                    prev[idx].visibility > 0.35f && curr[idx].visibility > 0.35f)
                {
                    rightMotion += Vector3.Distance(prev[idx].position, curr[idx].position);
                    rightCount++;
                }
            }
        }
        leftMotion = leftCount > 0 ? leftMotion / leftCount : 0f;
        rightMotion = rightCount > 0 ? rightMotion / rightCount : 0f;
        float total = leftMotion + rightMotion + 1e-6f;
        return Mathf.Abs(leftMotion - rightMotion) / total;
    }

    private float ComputeLegElevation(List<PoseLandmark> landmarks)
    {
        if (landmarks.Count < 28) return 0f;
        float leftHipY = landmarks[23].position.y;
        float rightHipY = landmarks[24].position.y;
        float leftAnkleY = landmarks[27].position.y;
        float rightAnkleY = landmarks[28].position.y;
        float torsoLength = Mathf.Abs(landmarks[11].position.y - landmarks[23].position.y);
        if (torsoLength < 0.01f) torsoLength = 0.1f;

        float leftElev = (leftHipY - leftAnkleY) / torsoLength;
        float rightElev = (rightHipY - rightAnkleY) / torsoLength;
        return Mathf.Max(0f, Mathf.Max(leftElev, rightElev));
    }

    private void StopWebcamAndLoadScene()
    {
        PoseLandmarkerRunner runner = FindObjectOfType<PoseLandmarkerRunner>();
        if (runner != null)
        {
            runner.Stop();
            Destroy(runner.gameObject);
            Debug.Log("CalibrationManager: Stopped PoseLandmarkerRunner and webcam.");
        }
        SceneManager.LoadScene(nextSceneName);
    }

    // ===== NEW METHOD – SKIP CURRENT GESTURE =====
    public void SkipCurrentGesture()
    {
        if (tutorialCompleted || currentStep >= steps.Count)
            return;

        // Reset current step progress
        steps[currentStep].currentHoldTime = 0f;
        steps[currentStep].currentCount = 0;
        steps[currentStep].completed = true;

        currentStep++;

        UpdateProgressUI();

        if (currentStep >= steps.Count)
        {
            tutorialCompleted = true;

            progressBarFill.fillAmount = 1f;
            progressText.text = $"Calibration {steps.Count}/{steps.Count}";

            statusText.text = "Calibration complete!";
            hintText.text = "Press Play or Jump to enter the game.";

            startGameButton.interactable = true;
            return;
        }

        CalibrationStep step = steps[currentStep];

        UpdateGestureIllustration(step.gestureName);

        statusText.text = GetStatusTextForStep(step);
        hintText.text = $"Hint:\n{step.instruction}";

        SetDetectionUI(notDetectedSprite, "NOT\nDETECTED", Color.red);
    }

    void EvaluateCalibration(List<PoseLandmark> landmarks,
                            float jumpImpulse,
                            float legMotion,
                            float legAsymmetry,
                            float legElevation)
    {
        if (!tutorialCompleted)
        {
            if (landmarks.Count < 29)
            {
                statusText.text = "Incomplete body tracking step back";
                startGameButton.interactable = false;
                return;
            }

            float avgVis = 0f;
            foreach (var lm in landmarks) avgVis += lm.visibility;
            avgVis /= landmarks.Count;

            bool feetVisible = landmarks[27].visibility > 0.5f && landmarks[28].visibility > 0.5f;
            bool wristsVisible = landmarks[15].visibility > 0.5f && landmarks[16].visibility > 0.5f;
            bool kneesVisible = landmarks[25].visibility > 0.5f && landmarks[26].visibility > 0.5f;
            float centerX = (landmarks[11].position.x + landmarks[12].position.x) * 0.5f;
            bool centered = centerX > 0.15f && centerX < 0.85f;

            bool headVisible = landmarks[0].visibility > 0.5f;

            UpdateBodyTrackingUI(
                headVisible,
                wristsVisible,
                kneesVisible,
                feetVisible
            );

            string feedback = "";
            if (avgVis < minVisibilityThreshold) feedback += "• Stand fully in frame\n";
            if (!feetVisible) feedback += "• Feet not visible step back or\n  lower camera\n";
            if (!wristsVisible) feedback += "• Keep both hands visible\n";
            if (!kneesVisible) feedback += "• Ensure knees are in view\n";
            if (!centered) feedback += "• Move to center\n";

            bool bodyOK = avgVis >= minVisibilityThreshold && feetVisible && wristsVisible && kneesVisible && centered;

            if (!bodyOK)
            {
                SetDetectionUI(notDetectedSprite, "TRACKING LOST", Color.red);
                statusText.text = "Adjust position";

                if (feedbackText != null)
                    feedbackText.text = feedback.TrimEnd('\n');
                if (hintText != null)
                    hintText.text = $"Hint:\n{(currentStep < steps.Count ? steps[currentStep].instruction : "All done!")}";

                startGameButton.interactable = false;
                return;
            }
            else
            {
                if (feedbackText != null)
                    feedbackText.text = "";
            }
        }

        if (currentStep >= steps.Count)
        {
            tutorialCompleted = true;

            if (progressText != null)
                progressText.text = $"Calibration {steps.Count}/{steps.Count}";
            if (progressBarFill != null)
                progressBarFill.fillAmount = 1f;

            SetDetectionUI(detectedSprite, "CALIBRATION COMPLETE", Color.green);

            statusText.text = "All checks passed!";
            startGameButton.interactable = true;
            hintText.text = "Calibration complete!\nPress 'Play' or JUMP to enter the Game.";

            if (feedbackText != null)
                feedbackText.text = "";

            return;
        }

        var step = steps[currentStep];
        UpdateGestureIllustration(step.gestureName);

        bool gestureMatched = currentGesture == step.gestureName && currentConfidence >= minGestureConfidence;

        bool kickDetected = false;
        if (step.gestureName == "kick")
        {
            bool hasLegTracking = (landmarks[25].visibility > 0.2f || landmarks[26].visibility > 0.2f) &&
                                  (landmarks[27].visibility > 0.2f || landmarks[28].visibility > 0.2f);
            bool legRaised = legElevation > KICK_ELEVATION_THRESHOLD;
            bool motionTrigger = hasLegTracking && legMotion >= KICK_MOTION_THRESHOLD && legAsymmetry >= KICK_ASYMMETRY_THRESHOLD;
            float leftAnkleY = landmarks[27].position.y;
            float rightAnkleY = landmarks[28].position.y;
            float ankleDiff = Mathf.Abs(leftAnkleY - rightAnkleY);
            float torsoLen = Mathf.Abs(landmarks[11].position.y - landmarks[23].position.y);
            if (torsoLen < 0.01f) torsoLen = 0.1f;
            bool ankleDifference = ankleDiff / torsoLen > 0.1f;
            bool notJumping = jumpImpulse < JUMP_IMPULSE_MAX_FOR_KICK;
            kickDetected = notJumping && (legRaised || motionTrigger || ankleDifference);
        }

        bool jumpDetected = false;
        if (step.gestureName == "jump" && !gestureMatched)
        {
            jumpDetected = jumpImpulse > 0.035f;
        }

        bool stepCompleted = false;
        if (step.gestureName == "jump")
            stepCompleted = gestureMatched || jumpDetected;
        else if (step.gestureName == "kick")
            stepCompleted = gestureMatched || kickDetected;
        else
            stepCompleted = gestureMatched;

        if (stepCompleted)
        {
            wrongGestureMessage = "";
            if (step.holdRequired)
            {
                step.currentHoldTime += Time.deltaTime;
                if (step.currentHoldTime >= holdRequiredDuration)
                {
                    step.completed = true;
                    currentStep++;
                    UpdateProgressUI();
                    SetDetectionUI(detectedSprite, "DETECTED", Color.green);
                    StartSafeGreenSplash();
                    if (currentStep < steps.Count)
                        statusText.text = GetStatusTextForStep(steps[currentStep]);
                    else
                        statusText.text = "All checks passed!";
                }
                else
                {
                    float remaining = holdRequiredDuration - step.currentHoldTime;
                    SetDetectionUI(holdingSprite, $"HOLD {remaining:F1}s", Color.yellow);
                    statusText.text = (step.gestureName == "idle") ? "Stay still" : "Hold still";
                }
            }
            else
            {
                step.currentCount++;
                if (step.currentCount >= repeatRequiredCount)
                {
                    step.completed = true;
                    currentStep++;
                    UpdateProgressUI();
                    SetDetectionUI(detectedSprite, "DETECTED", Color.green);
                    StartSafeGreenSplash();
                    if (currentStep < steps.Count)
                        statusText.text = GetStatusTextForStep(steps[currentStep]);
                    else
                        statusText.text = "All checks passed!";
                }
                else
                {
                    statusText.text = $"{FormatGestureName(step.gestureName)} {step.currentCount}/{repeatRequiredCount} – keep going!";
                }
            }
        }
        else if (currentGesture != "idle" && currentGesture != step.gestureName && step.gestureName != "kick")
        {
            wrongGestureMessage = GetStatusTextForStep(step);
            wrongGestureTimer = 1f;
            step.currentHoldTime = 0f;
            step.currentCount = 0;
            SetDetectionUI(notDetectedSprite, "WRONG GESTURE", Color.red);
        }
        else
        {
            SetDetectionUI(notDetectedSprite, "NOT DETECTED", Color.red);
            statusText.text = GetStatusTextForStep(step);
        }

        string hint = step.instruction;
        if (hintText != null)
            hintText.text = $"Hint:\n{hint}";

        startGameButton.interactable = false;
    }

    void UpdateWrongGestureMessage()
    {
        if (wrongGestureTimer > 0)
        {
            wrongGestureTimer -= Time.deltaTime;
            if (wrongGestureTimer <= 0)
                wrongGestureMessage = "";
        }
        if (!string.IsNullOrEmpty(wrongGestureMessage))
        {
            statusText.text = wrongGestureMessage;
        }
    }

    private string GetStatusTextForStep(CalibrationStep step)
    {
        if (step.gestureName == "idle")
            return "Stand naturally";
        else
            return $"Perform {FormatGestureName(step.gestureName)}";
    }

    // --- UPDATED FormatGestureName to show "TURN LEFT" / "TURN RIGHT" ---
    private string FormatGestureName(string gesture)
    {
        switch (gesture)
        {
            case "lean_left":
                return "TURN LEFT";

            case "lean_right":
                return "TURN RIGHT";

            case "run":
                return "RUN";

            case "jump":
                return "JUMP";

            case "punch":
                return "PUNCH";

            case "kick":
                return "KICK";

            case "idle":
                return "IDLE";

            default:
                return gesture.ToUpper();
        }
    }

    private void UpdateGestureIllustration(string gesture)
    {
        if (gestureIllustration == null)
            return;

        switch (gesture)
        {
            case "idle":
                gestureIllustration.sprite = idleSprite;
                break;
            case "run":
                gestureIllustration.sprite = runSprite;
                break;
            case "lean_left":
                gestureIllustration.sprite = leanLeftSprite;
                break;
            case "lean_right":
                gestureIllustration.sprite = leanRightSprite;
                break;
            case "jump":
                gestureIllustration.sprite = jumpSprite;
                break;
            case "punch":
                gestureIllustration.sprite = punchSprite;
                break;
            case "kick":
                gestureIllustration.sprite = kickSprite;
                break;
        }
    }

    private void SetDetectionUI(Sprite icon, string text, Color color)
    {
        if (detectionImage != null)
            detectionImage.sprite = icon;

        if (detectionText != null)
        {
            detectionText.text = text;
            detectionText.color = color;
        }
    }

    private void UpdateBodyTrackingUI(
        bool headVisible,
        bool handsVisible,
        bool kneesVisible,
        bool feetVisible)
    {
        if (headDot != null)
            headDot.sprite = headVisible ? greenDotSprite : redDotSprite;

        if (handsDot != null)
            handsDot.sprite = handsVisible ? greenDotSprite : redDotSprite;

        if (kneesDot != null)
            kneesDot.sprite = kneesVisible ? greenDotSprite : redDotSprite;

        if (feetDot != null)
            feetDot.sprite = feetVisible ? greenDotSprite : redDotSprite;
    }

    private void StartSafeGreenSplash()
    {
        if (splashRoutine != null)
            StopCoroutine(splashRoutine);

        splashRoutine = StartCoroutine(GreenSplash());
    }

    private IEnumerator GreenSplash()
    {
        if (successSplashImage == null)
            yield break;

        successSplashImage.gameObject.SetActive(true);
        Color c = successSplashImage.color;
        c.a = 1f;
        successSplashImage.color = c;

        float duration = 0.3f;
        float timer = 0f;
        while (timer < duration)
        {
            timer += Time.deltaTime;
            c.a = Mathf.Lerp(1f, 0f, timer / duration);
            successSplashImage.color = c;
            yield return null;
        }

        c.a = 0f;
        successSplashImage.color = c;
        successSplashImage.gameObject.SetActive(false);

        splashRoutine = null;
    }

    private void UpdateProgressUI()
    {
        int totalSteps = steps.Count;
        int displayStep = Mathf.Min(currentStep + 1, totalSteps);

        if (progressText != null)
            progressText.text = $"Calibration {displayStep}/{totalSteps}";

        float progress = (float)currentStep / totalSteps;

        if (progressBarFill != null)
            progressBarFill.fillAmount = progress;
    }
}

public class CalibrationStep
{
    public string gestureName;
    public string instruction;
    public bool holdRequired;
    public int requiredCount;
    public float currentHoldTime;
    public int currentCount;
    public bool completed;

    public CalibrationStep(string name, string instr, bool holdRequired, int requiredCount)
    {
        this.gestureName = name;
        this.instruction = instr;
        this.holdRequired = holdRequired;
        this.requiredCount = requiredCount;
        this.currentHoldTime = 0f;
        this.currentCount = 0;
        this.completed = false;
    }
}
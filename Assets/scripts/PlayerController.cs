using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class PlayerController : MonoBehaviour
{
    [Header("Movement Settings")]
    public float speed = 5f;
    public float jumpHeight = 1.5f;
    public float gravity = -9.8f;
    public float squatDuration = 0.5f;

    [Header("Water Settings")]
    public float waterSinkSpeed = 1.2f;
    public float waterMoveSpeedMultiplier = 0.4f;

    [Header("Smooth Turning Settings")]
    public float smoothTurnTime = 0.15f;
    public float turnAmountGesture = 45f;

    [Header("Run Settings")]
    public float runSignalTimeout = 0.05f;
    public float runLockActivationTime = 5f;

    [Header("Camera Reference")]
    public Transform cameraTransform;

    [Header("Gesture Confidence")]
    public float runConfidenceReq = 0.10f;
    public float jumpConfidenceReq = 0.12f;
    public float punchConfidenceReq = 0.16f;
    public float kickConfidenceReq = 0.12f;
    public float blockConfidenceReq = 0.15f;
    public float dodgeConfidenceReq = 0.12f;
    public float leanConfidenceReq = 0.12f;

    [Header("Combat")]
    public CombatController combatController;

    [Header("Debug")]
    public bool logActions = true;

    public bool IsRunLocked => isRunLocked;

    private PlayerStats playerStats;
    private CharacterController controller;
    private Animator animator;
    private Vector3 velocity;

    private float targetYaw;
    private float currentYaw;
    private float yawVelocity;

    private bool wasGrounded;
    private bool isJumping;
    private bool isAttacking;
    private bool isGuarding;
    private bool isRunning;
    private float lastRunSignalTime;
    private bool runDrivenByGesture;
    private bool runStartedByStaticPose;

    private bool gestureRunActive = false;
    private float runTimer;
    private bool isRunLocked;

    private Coroutine squatCoroutine;
    private Dictionary<string, System.Action> gestureActionMap;
    private string currentActiveGesture = "";

    // --- Lean auto-reset fields ---
    private float leanHoldTimer = 0f;
    private string lockedLeanGesture = "";
    private float leanResetTime = 3f;
    // -----------------------------------

    private RuleBasedGestureRecognizer gestureRecognizer;
    private float lastRecognizerTime;
    private float lastLogTime = 0f;

    // --- Footstep cycling ---
    private int footstepIndex = 0;

    void Start()
    {
        controller = GetComponent<CharacterController>();
        animator = GetComponent<Animator>();
        playerStats = GetComponent<PlayerStats>();

        if (combatController == null)
            combatController = GetComponent<CombatController>();

        if (cameraTransform == null)
        {
            Camera mainCam = Camera.main;
            if (mainCam != null) cameraTransform = mainCam.transform;
        }

        wasGrounded = controller.isGrounded;
        currentYaw = transform.eulerAngles.y;
        targetYaw = currentYaw;

        InitializeGestureMap();

        gestureRecognizer = new RuleBasedGestureRecognizer();
        lastRecognizerTime = Time.time;

        Debug.Log("[PlayerController] Initialized. MediaPipePoseManager.Instance is " + (MediaPipePoseManager.Instance != null ? "present" : "NULL"));
    }

    void InitializeGestureMap()
    {
        gestureActionMap = new Dictionary<string, System.Action>
        {
            { "run",         () => StartRunningFromGesture() },
            { "run_start",   () => StartRunningFromStaticPose() },
            { "run_stop",    () => StopRunning() },
            { "jump",        () => Jump() },
            { "punch",       () => TriggerAttack() },
            { "attack",      () => TriggerAttack() },
            { "kick",        () => TriggerKick() },
            { "block",       () => StartBlocking() },
            { "block_start", () => StartBlocking() },
            { "block_stop",  () => StopBlocking() },
            { "dodge",       () => Squat() },
            { "lean_left",   () => QueueTurn(-1) },
            { "lean_right",  () => QueueTurn(1) },
            { "idle",        () => StopAllActions() },
            { "stop",        () => StopAllActions() }
        };
    }

    void Update()
    {
        if (playerStats != null && playerStats.isDead) return;
        if (PauseMenu.isPaused) return;
        if (controller == null) return;

        // Process landmarks
        List<PoseLandmark> landmarks = MediaPipePoseManager.Instance?.GetCurrentLandmarks();
        
        if (Time.time - lastLogTime > 3f)
        {
            lastLogTime = Time.time;
            if (landmarks == null)
                Debug.LogWarning("[PlayerController] Landmarks are NULL");
            else if (landmarks.Count < 33)
                Debug.LogWarning($"[PlayerController] Landmarks count: {landmarks.Count} (need 33)");
            else
                Debug.Log($"[PlayerController] Landmarks OK: {landmarks.Count} landmarks received");
        }

        if (landmarks != null && landmarks.Count >= 33)
        {
            GestureResult result = gestureRecognizer.ProcessFrame(landmarks, Time.deltaTime);
            if (result != null && !string.IsNullOrEmpty(result.gesture) && !result.cooldownBlocked)
            {
                if (logActions) Debug.Log($"[PlayerController] Recognized gesture: {result.gesture} (conf: {result.confidence:F2})");
                HandleGesture(result.gesture, result.confidence);
            }
            else if (result != null && result.cooldownBlocked)
            {
                if (logActions) Debug.Log($"[PlayerController] Gesture {result.gesture} blocked by cooldown");
            }
        }

        // Run-lock timer
        if (isRunning)
            runTimer += Time.deltaTime;
        else
            runTimer = 0f;

        if (!isRunLocked && runTimer >= runLockActivationTime)
        {
            isRunLocked = true;
            if (logActions) Debug.Log("RUN LOCK ACTIVATED");
        }

        if (isRunLocked && !isRunning)
        {
            isRunLocked = false;
            runTimer = 0f;
        }

        if (Mathf.Abs(targetYaw - currentYaw) > 0.01f)
        {
            currentYaw = Mathf.SmoothDampAngle(currentYaw, targetYaw, ref yawVelocity, smoothTurnTime);
            transform.rotation = Quaternion.Euler(0, currentYaw, 0);
        }

        HandleKeyboardInput();
        UpdateAttackState();
        HandleGravity();
        HandleMovement();
        UpdateAnimator();
        CheckGrounded();

        // --- Lean hold reset ---
        UpdateLeanReset();
        // ---------------------------

        if (!isRunLocked && isRunning && !gestureRunActive && runDrivenByGesture && !runStartedByStaticPose &&
            Time.time - lastRunSignalTime > runSignalTimeout)
        {
            StopRunning();
        }
    }

    #region Gesture Handling
    public void HandleGesture(string gesture, float confidence)
    {
        if (logActions) Debug.Log($"[PlayerController] HandleGesture called: {gesture} (conf: {confidence:F2})");

        if (playerStats != null && playerStats.isDead) return;
        if (string.IsNullOrEmpty(gesture)) return;

        if ((gesture == "idle" || gesture == "stop") && isRunLocked)
        {
            if (logActions) Debug.Log($"[PlayerController] Run lock – ignoring {gesture}");
            currentActiveGesture = gesture;
            return;
        }

        if (gesture == "idle" || gesture == "stop")
        {
            StopAllActions();
            currentActiveGesture = gesture;
            return;
        }

        float required = GetConfidenceFor(gesture);
        if (confidence < required)
        {
            if (logActions) Debug.Log($"[PlayerController] Ignoring {gesture} (conf {confidence:F2} < {required:F2})");
            return;
        }

        if (isRunLocked && gesture != "run" && gesture != "run_start"
            && gesture != "run_stop" && gesture != "lean_left"
            && gesture != "lean_right" && gesture != "jump")
        {
            CancelRunLockAndStop();
            return;
        }

        if (isRunning && gesture != "run" && gesture != "run_start"
            && gesture != "run_stop" && gesture != "lean_left"
            && gesture != "lean_right" && gesture != "jump")
            StopRunning();

        if (gestureActionMap.ContainsKey(gesture))
        {
            if (gesture == "lean_left" || gesture == "lean_right")
            {
                if (gesture != currentActiveGesture)
                {
                    if (logActions) Debug.Log($"[PlayerController] Executing One-Shot Turn: {gesture}");
                    gestureActionMap[gesture]?.Invoke();
                }
            }
            else
            {
                if (logActions) Debug.Log($"[PlayerController] Executing: {gesture}");
                gestureActionMap[gesture]?.Invoke();
                if (gesture == "run") RefreshRunSignal();
            }
        }
        else
        {
            if (logActions) Debug.LogWarning($"[PlayerController] No action mapped for gesture: {gesture}");
        }

        currentActiveGesture = gesture;
    }

    private float GetConfidenceFor(string gesture)
    {
        switch (gesture)
        {
            case "run": return runConfidenceReq;
            case "jump": return jumpConfidenceReq;
            case "punch":
            case "attack": return punchConfidenceReq;
            case "kick": return kickConfidenceReq;
            case "block":
            case "block_start":
            case "block_stop": return blockConfidenceReq;
            case "dodge": return dodgeConfidenceReq;
            case "lean_left":
            case "lean_right": return leanConfidenceReq;
            default: return 0f;
        }
    }
    #endregion

    #region Movement & Actions
    private void QueueTurn(int direction)
    {
        if (isAttacking || isGuarding) return;
        targetYaw += turnAmountGesture * direction;
        if (logActions) Debug.Log($"[PlayerController] Queue smooth turn: {(direction < 0 ? "Left" : "Right")}");
    }

    private void StartRunningFromGesture() { if (!isGuarding) { runDrivenByGesture = true; runStartedByStaticPose = false; lastRunSignalTime = Time.time; gestureRunActive = true; StartRunning(); } }
    private void StartRunningFromStaticPose() { if (!isGuarding) { runDrivenByGesture = true; runStartedByStaticPose = true; lastRunSignalTime = Time.time; gestureRunActive = true; StartRunning(); } }

    private void StartRunning()
    {
        if (!isRunning && !isAttacking && !isGuarding)
        {
            if (logActions) Debug.Log("[PlayerController] Start running");
            isRunning = true;
            if (animator != null) animator.SetBool("isRunning", true);
        }
    }

    private void StopRunning()
    {
        if (isRunning)
        {
            if (logActions) Debug.Log("[PlayerController] Stop running");
            isRunning = false;
            if (animator != null) animator.SetBool("isRunning", false);
        }
        gestureRunActive = false;
        isRunLocked = false;
        runTimer = 0f;
        runDrivenByGesture = false;
        runStartedByStaticPose = false;
    }

    private void CancelRunLockAndStop()
    {
        isRunLocked = false;
        runTimer = 0f;
        StopRunning();
    }

    private void RefreshRunSignal() { runDrivenByGesture = true; lastRunSignalTime = Time.time; }

    private void Jump()
    {
        if (playerStats != null && playerStats.isInWater) return;
        if (isJumping || !wasGrounded || isAttacking || isGuarding) return;
        float jumpVelocity = Mathf.Sqrt(jumpHeight * -2f * gravity);
        velocity.y = jumpVelocity;
        isJumping = true;
        if (animator != null) animator.SetTrigger("Jump");
        if (logActions) Debug.Log("[PlayerController] Jump executed");
    }

    private void StartBlocking()
    {
        if (isAttacking) return;
        if (!isGuarding)
        {
            if (logActions) Debug.Log("[PlayerController] Start blocking");
            isGuarding = true;
            CancelRunLockAndStop();
            if (animator != null) animator.SetBool("Guard", true);
        }
    }

    public void StopBlocking()
    {
        if (!isGuarding) return;
        if (logActions) Debug.Log("[PlayerController] Stop blocking");
        isGuarding = false;
        if (animator != null) animator.SetBool("Guard", false);
    }

    private void TriggerAttack()
    {
        if (isGuarding || combatController == null) return;
        CancelRunLockAndStop();
        if (logActions) Debug.Log("[PlayerController] TriggerAttack → CombatController");
        combatController.Punch();
    }

    private void TriggerKick()
    {
        if (isGuarding || combatController == null) return;
        CancelRunLockAndStop();
        if (logActions) Debug.Log("[PlayerController] TriggerKick → CombatController");
        combatController.Kick();
    }

    private void Squat()
    {
        if (isAttacking || isGuarding) return;
        CancelRunLockAndStop();
        if (logActions) Debug.Log("[PlayerController] Dodge/Squat");
        if (animator != null) animator.SetTrigger("Dodge");
        if (squatCoroutine != null) StopCoroutine(squatCoroutine);
        squatCoroutine = StartCoroutine(SquatRoutine());
    }

    private IEnumerator SquatRoutine()
    {
        if (controller != null) { controller.height = 1.0f; controller.center = new Vector3(0, 0.5f, 0); }
        yield return new WaitForSeconds(squatDuration);
        if (controller != null) { controller.height = 2.0f; controller.center = new Vector3(0, 1.0f, 0); }
    }

    private void StopAllActions()
    {
        if (logActions) Debug.Log("[PlayerController] Stop all actions");
        StopRunning();
        StopBlocking();
        if (squatCoroutine != null) { StopCoroutine(squatCoroutine); squatCoroutine = null; }
        if (controller != null) { controller.height = 2.0f; controller.center = new Vector3(0, 1.0f, 0); }
    }
    #endregion

    #region Core Physics & Movement
    private void HandleMovement()
    {
        if (isAttacking || isGuarding)
        {
            controller.Move(new Vector3(0, velocity.y, 0) * Time.deltaTime);
            return;
        }

        if (playerStats != null && playerStats.isInWater)
        {
            Vector3 drownMove = Vector3.down * waterSinkSpeed;
            controller.Move(drownMove * Time.deltaTime);
            return;
        }

        Vector3 move = Vector3.zero;
        if (isRunning) move = transform.forward * speed;
        move.y = velocity.y;
        controller.Move(move * Time.deltaTime);
    }

    private void HandleKeyboardInput()
    {
        float horizontal = 0f, vertical = 0f;
        if (Input.GetKey(KeyCode.UpArrow) || Input.GetKey(KeyCode.W)) vertical = 1f;
        if (Input.GetKey(KeyCode.DownArrow) || Input.GetKey(KeyCode.S)) vertical = -1f;
        if (Input.GetKey(KeyCode.LeftArrow) || Input.GetKey(KeyCode.A)) horizontal = -1f;
        if (Input.GetKey(KeyCode.RightArrow) || Input.GetKey(KeyCode.D)) horizontal = 1f;

        bool isMoving = (horizontal != 0 || vertical != 0);

        if (!isAttacking && !isGuarding && isMoving)
        {
            Vector3 moveDir = GetCameraRelativeMovement(horizontal, vertical);
            if (moveDir.magnitude > 0.1f)
            {
                Quaternion targetRot = Quaternion.LookRotation(moveDir);
                transform.rotation = Quaternion.Slerp(transform.rotation, targetRot, 10f * Time.deltaTime);
                currentYaw = transform.eulerAngles.y;
                targetYaw = currentYaw;
            }
            StartRunning();
        }
        else if (!isMoving && !runDrivenByGesture && !isRunLocked)
        {
            StopRunning();
        }

        if (Input.GetKeyDown(KeyCode.LeftShift)) Squat();
        if (Input.GetKeyDown(KeyCode.Q)) QueueTurn(-1);
        if (Input.GetKeyDown(KeyCode.E)) QueueTurn(1);
        if (Input.GetKeyDown(KeyCode.Space)) Jump();
        if (Input.GetKey(KeyCode.G)) StartBlocking();
        else if (Input.GetKeyUp(KeyCode.G)) StopBlocking();

        if (!isAttacking)
        {
            if (Input.GetMouseButtonDown(0)) TriggerAttack();
            if (Input.GetMouseButtonDown(1)) TriggerKick();
        }
    }

    private Vector3 GetCameraRelativeMovement(float horizontal, float vertical)
    {
        if (cameraTransform == null) return new Vector3(horizontal, 0, vertical);
        Vector3 forward = cameraTransform.forward; forward.y = 0; forward.Normalize();
        Vector3 right = cameraTransform.right; right.y = 0; right.Normalize();
        return (forward * vertical) + (right * horizontal);
    }

    private void HandleGravity()
    {
        if (playerStats != null && playerStats.isInWater)
        {
            velocity.y = 0;
            return;
        }
        if (controller.isGrounded && velocity.y < 0)
            velocity.y = -2f;
        else
            velocity.y += gravity * Time.deltaTime;
    }

    private void CheckGrounded()
    {
        bool currentGrounded = controller.isGrounded;
        if (currentGrounded && !wasGrounded) isJumping = false;
        wasGrounded = currentGrounded;
        if (animator != null) animator.SetBool("isGrounded", currentGrounded);
    }

    private void UpdateAttackState()
    {
        if (animator == null) return;
        AnimatorStateInfo stateInfo = animator.GetCurrentAnimatorStateInfo(0);
        isAttacking = stateInfo.IsTag("Attack");
    }

    private void UpdateAnimator()
    {
        if (animator == null) return;
        animator.SetBool("isGrounded", controller.isGrounded);
        animator.SetBool("isRunning", isRunning);
    }
    #endregion

    // --- Lean auto‑reset method ---
    private void UpdateLeanReset()
    {
        if (currentActiveGesture == "lean_left" ||
            currentActiveGesture == "lean_right")
        {
            leanHoldTimer += Time.deltaTime;

            if (leanHoldTimer >= leanResetTime)
            {
                if (logActions)
                    Debug.Log("Lean gesture timeout reset");

                currentActiveGesture = "";
                leanHoldTimer = 0f;
            }
        }
        else
        {
            leanHoldTimer = 0f;
        }
    }
    // -----------------------------------

    #region Animation Events

    [Header("Audio")]
    public AudioClip[] footstepSounds;          // Array of footstep clips
    [Range(0f, 1f)]
    public float footstepVolume = 0.5f;

    /// <summary>
    /// Called by animation events on the Run cycle.
    /// Cycles through the footstepSounds array to play a different clip each step.
    /// </summary>
    public void OnFootstep()
    {
        if (footstepSounds == null || footstepSounds.Length == 0)
            return;

        AudioManager.Instance.PlaySFX(footstepSounds[footstepIndex], footstepVolume);

        footstepIndex++;
        if (footstepIndex >= footstepSounds.Length)
            footstepIndex = 0;
    }

    public void OnJumpComplete() { isJumping = false; }

    // Placeholder for future attack impact sounds
    public void OnAttackComplete() { }

    #endregion
}
using UnityEngine;

public class PlayerMovement : MonoBehaviour
{
    public float speed = 5f;
    public float jumpHeight = 1.5f;        // Height of the jump
    public float gravity = -9.8f;
    public float rotationSpeed = 10f;
    public float groundCheckBuffer = 0.1f;  // Buffer time for ground check

    // Reference to the camera (can be assigned in inspector or found automatically)
    public Transform cameraTransform;

    private CharacterController controller;
    private Animator animator;
    private Vector3 velocity;
    private bool wasGrounded;           // Track previous grounded state for transition
    private float groundTimer;          // Timer for ground check buffer
    private bool isJumping;             // Prevent multiple jump triggers
    private bool isAttacking;           // Lock movement during attacks
    private bool isGuarding;            // Lock movement during guard

    void Start()
    {
        controller = GetComponent<CharacterController>();
        animator = GetComponent<Animator>();

        // If camera transform is not assigned, try to find the main camera
        if (cameraTransform == null)
        {
            Camera mainCamera = Camera.main;
            if (mainCamera != null)
            {
                cameraTransform = mainCamera.transform;
            }
        }

        wasGrounded = controller.isGrounded;
        groundTimer = 0f;
        isJumping = false;
        isAttacking = false;
        isGuarding = false;
    }

    void Update()
    {
        // Check attack states using tags
        UpdateAttackState();

        // Get input
        float horizontal = 0f;
        float vertical = 0f;

        if (Input.GetKey(KeyCode.UpArrow)) vertical = 1f;
        if (Input.GetKey(KeyCode.DownArrow)) vertical = -1f;
        if (Input.GetKey(KeyCode.LeftArrow)) horizontal = -1f;
        if (Input.GetKey(KeyCode.RightArrow)) horizontal = 1f;

        // Handle combat actions
        HandleCombat();

        // Handle jumping
        HandleJump();

        // Handle gravity
        HandleGravity();

        // Handle movement and apply combined move
        HandleMovement(horizontal, vertical);

        // Update animator parameters (isRunning, LeanLeft, LeanRight)
        UpdateAnimator(horizontal, vertical);

        // Update grounded state for next frame
        bool currentGrounded = controller.isGrounded;

        // Update ground timer for buffer
        if (currentGrounded)
        {
            groundTimer = groundCheckBuffer;
        }
        else
        {
            groundTimer -= Time.deltaTime;
        }

        wasGrounded = currentGrounded;
    }

    void HandleMovement(float horizontal, float vertical)
    {
        Vector3 finalMove;

        // Lock movement during attacks OR guard
        if (isAttacking || isGuarding)
        {
            // During attacks or guard: no horizontal movement, only vertical (gravity)
            finalMove = new Vector3(0, velocity.y, 0);
            controller.Move(finalMove * Time.deltaTime);
            return;
        }

        // Get camera-relative movement directions
        Vector3 move = GetCameraRelativeMovement(horizontal, vertical);

        // Normalize to maintain consistent speed in diagonal movement
        if (move.magnitude > 0.1f)
        {
            move.Normalize();
        }

        // Combine horizontal movement with vertical velocity
        finalMove = move * speed;
        finalMove.y = velocity.y;

        // Single Move call per frame
        controller.Move(finalMove * Time.deltaTime);

        // Rotate toward movement direction (only when moving on ground)
        if (move.magnitude > 0.1f && controller.isGrounded)
        {
            Quaternion targetRotation = Quaternion.LookRotation(move);
            transform.rotation = Quaternion.Slerp(
                transform.rotation,
                targetRotation,
                rotationSpeed * Time.deltaTime
            );
        }
    }

    void HandleJump()
    {
        // Don't jump while attacking or guarding
        if (isAttacking || isGuarding) return;

        // Use wasGrounded buffer to prevent missed jumps
        if ((controller.isGrounded || wasGrounded) && Input.GetKeyDown(KeyCode.Space) && !isJumping)
        {
            // Calculate jump velocity using physics formula: v = sqrt(2 * gravity * height)
            float jumpVelocity = Mathf.Sqrt(jumpHeight * -2f * gravity);
            velocity.y = jumpVelocity;

            // Set jumping flag to prevent multiple jumps
            isJumping = true;

            // Trigger jump animation
            if (animator != null)
            {
                animator.SetTrigger("Jump");
            }
        }

        // Reset jumping flag when grounded again
        if (controller.isGrounded && velocity.y <= 0f)
        {
            isJumping = false;
        }
    }

    void HandleCombat()
    {
        if (animator == null) return;

        // Guard (bool) - with movement lock
        if (Input.GetKey(KeyCode.G))
        {
            if (!isGuarding)
            {
                isGuarding = true;
                animator.SetBool("Guard", true);
            }
            return; // Stop processing attacks while guarding
        }
        else
        {
            if (isGuarding)
            {
                isGuarding = false;
                animator.SetBool("Guard", false);
            }
        }

        // Attacks (triggers) - can be performed anytime except during other attacks
        if (!isAttacking)
        {
            if (Input.GetKeyDown(KeyCode.Mouse0))
            {
                animator.SetTrigger("Attack");  // Punch
            }

            if (Input.GetKeyDown(KeyCode.Mouse1))
            {
                animator.SetTrigger("Kick");
            }

            if (Input.GetKeyDown(KeyCode.LeftShift))
            {
                animator.SetTrigger("Dodge");
            }
        }
    }

    void UpdateAttackState()
    {
        if (animator == null) return;

        // Use tags to detect attack states
        AnimatorStateInfo stateInfo = animator.GetCurrentAnimatorStateInfo(0);

        // Check if current state has "Attack" or "Dodge" tag
        isAttacking = stateInfo.IsTag("Attack") || stateInfo.IsTag("Dodge");

        // Guard state is handled separately by isGuarding flag
    }

    void HandleGravity()
    {
        // Simplified gravity handling
        if (controller.isGrounded)
        {
            if (velocity.y < 0)
                velocity.y = -2f;
        }
        else
        {
            velocity.y += gravity * Time.deltaTime;
        }
    }

    void UpdateAnimator(float horizontal, float vertical)
    {
        if (animator == null) return;

        // Always update grounded state
        animator.SetBool("isGrounded", controller.isGrounded);

        // --- Movement calculations for animator (speed parameter removed) ---
        Vector3 move = GetCameraRelativeMovement(horizontal, vertical);
        float currentSpeed = move.magnitude;

        // Determine if player is running (moving on ground and not locked by combat)
        bool isRunning = false;
        bool canMoveFreely = !isAttacking && !isGuarding && controller.isGrounded;

        if (canMoveFreely && currentSpeed > 0.1f)
        {
            isRunning = true;
        }
        else
        {
            isRunning = false;
        }

        // Only set isRunning (speed parameter is no longer used)
        animator.SetBool("isRunning", isRunning);

        // --- Leaning left/right based on horizontal input ---
        // Lean animations only play when running on ground
        bool canLean = canMoveFreely && isRunning;
        if (canLean)
        {
            // Raw horizontal input determines lean direction
            if (horizontal < -0.1f)
            {
                animator.SetBool("LeanLeft", true);
                animator.SetBool("LeanRight", false);
            }
            else if (horizontal > 0.1f)
            {
                animator.SetBool("LeanLeft", false);
                animator.SetBool("LeanRight", true);
            }
            else
            {
                animator.SetBool("LeanLeft", false);
                animator.SetBool("LeanRight", false);
            }
        }
        else
        {
            // Reset leaning when not running or in combat/jump
            animator.SetBool("LeanLeft", false);
            animator.SetBool("LeanRight", false);
        }
    }

    // Get movement direction relative to camera orientation
    private Vector3 GetCameraRelativeMovement(float horizontal, float vertical)
    {
        if (cameraTransform == null)
            return new Vector3(horizontal, 0, vertical);

        // Get camera's forward and right directions, ignoring pitch (keep horizontal plane)
        Vector3 cameraForward = cameraTransform.forward;
        Vector3 cameraRight = cameraTransform.right;

        // Remove vertical component to keep movement on ground plane
        cameraForward.y = 0;
        cameraRight.y = 0;

        // Normalize to maintain direction consistency
        cameraForward.Normalize();
        cameraRight.Normalize();

        // Calculate movement relative to camera
        Vector3 move = (cameraForward * vertical) + (cameraRight * horizontal);

        return move;
    }

    // Animation Event receiver for footstep sounds
    public void OnFootstep()
    {
        // Optional: You can add footstep sound logic here if you don't want to use the FootstepHandler
        Debug.Log("Footstep played - you can play sound here");
    }

    // Animation Event to signal jump animation is complete
    public void OnJumpComplete()
    {
        isJumping = false;
    }

    // Animation Event for attack complete (optional, for precise timing)
    public void OnAttackComplete()
    {
        // This can be used if you need to know exactly when attack animation finishes
        // Currently handled by state tags automatically
    }
}
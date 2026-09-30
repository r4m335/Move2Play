// MovementController.cs
using UnityEngine;

/**
 * MovementController.cs
 * 
 * Handles locomotion and posture:
 * - Run, walk, squat
 * - Lean left/right
 * - Jump, slide
 * - CharacterController movement
 */
public class MovementController : MonoBehaviour
{
    [Header("References")]
    public Animator animator;
    public CharacterController characterController;
    public Transform cameraToRotate;
    
    [Header("Movement Settings")]
    public float walkSpeed = 2.0f;
    public float runSpeed = 5.0f;
    public float crouchSpeed = 1.5f;
    public float turnSpeed = 90f;  // Degrees per second
    public float jumpHeight = 1.0f;
    public float gravity = -9.81f;
    public float slideDuration = 1.0f;
    public float slideSpeed = 8.0f;

    [Header("Lean Settings")]
    public float leanStepDegrees = 8.0f;
    public bool leanUseLocalRotation = true;
    
    [Header("Debug")]
    public bool logMovement = true;

    [Header("Run Signal")]
    public float runSignalTimeout = 0.45f;
    
    // State tracking
    private bool isRunning = false;
    private bool isWalking = false;
    private bool isCrouching = false;
    private bool isSliding = false;
    private float slideEndTime = 0f;
    private float verticalVelocity = 0f;
    private float currentSpeed = 0f;
    private bool isGrounded = true;
    private float lastRunSignalTime = -999f;
    private bool runDrivenByGesture = false;
    
    // Input smoothing
    private Vector3 moveDirection = Vector3.zero;
    
    void Awake()
    {
        if (animator == null)
            animator = GetComponent<Animator>();
            
        if (characterController == null)
            characterController = GetComponent<CharacterController>();

        if (cameraToRotate == null && Camera.main != null)
            cameraToRotate = Camera.main.transform;
    }
    
    void Start()
    {
        currentSpeed = walkSpeed;
    }
    
    void Update()
    {
        CheckGrounded();

        // Stop running if no run signal has been received recently.
        if (isRunning && runDrivenByGesture && Time.time - lastRunSignalTime > runSignalTimeout)
        {
            StopRunning();
        }
        
        // Handle continuous movement
        if (isRunning || isWalking)
        {
            MoveForward();
        }
        
        // Handle slide duration
        if (isSliding && Time.time >= slideEndTime)
        {
            StopSlide();
        }
        
        // Apply gravity
        ApplyGravity();
    }
    
    #region Public Action Methods
    
    public void StartRunning()
    {
        if (!isRunning && !isSliding)
        {
            if (logMovement) Debug.Log("[Movement] Start running");
            
            isRunning = true;
            isWalking = false;
            currentSpeed = runSpeed;
            
            animator.SetBool("isRunning", true);
            animator.SetBool("isWalking", false);
        }
    }

    public void StartRunningFromGesture()
    {
        runDrivenByGesture = true;
        lastRunSignalTime = Time.time;
        StartRunning();
    }
    
    public void StopRunning()
    {
        if (isRunning)
        {
            if (logMovement) Debug.Log("[Movement] Stop running");
            
            isRunning = false;
            isWalking = false;
            currentSpeed = walkSpeed;
            
            animator.SetBool("isRunning", false);
            animator.SetBool("isWalking", false);
            animator.SetTrigger("Idle");
        }

        runDrivenByGesture = false;
    }
    
    public void Walk()
    {
        if (!isWalking && !isSliding)
        {
            if (logMovement) Debug.Log("[Movement] Walk");
            
            isWalking = true;
            isRunning = false;
            currentSpeed = walkSpeed;
            
            animator.SetBool("isWalking", true);
            animator.SetBool("isRunning", false);
        }
    }
    
    public void Squat()
    {
        if (isSliding) return;
        
        if (!isCrouching)
        {
            if (logMovement) Debug.Log("[Movement] Squat/Crouch");
            
            isCrouching = true;
            currentSpeed = crouchSpeed;
            
            animator.SetTrigger("Squat");
            animator.SetBool("isCrouching", true);
            
            // Adjust character controller height for crouch
            if (characterController != null)
            {
                characterController.height = 1.0f; // Half height
                characterController.center = new Vector3(0, 0.5f, 0);
            }
        }
        else
        {
            // Stand up
            StandUp();
        }
    }
    
    public void StandUp()
    {
        if (!isCrouching) return;
        
        if (logMovement) Debug.Log("[Movement] Stand up");
        
        isCrouching = false;
        currentSpeed = isRunning ? runSpeed : walkSpeed;
        
        animator.SetBool("isCrouching", false);
        
        // Restore character controller height
        if (characterController != null)
        {
            characterController.height = 2.0f;
            characterController.center = new Vector3(0, 1.0f, 0);
        }
    }
    
    public void LeanLeft()
    {
        if (logMovement) Debug.Log($"[Movement] Lean left ({leanStepDegrees:F1} deg)");

        ApplyLeanRotation(-leanStepDegrees);
        animator.SetTrigger("LeanLeft");
    }
    
    public void LeanRight()
    {
        if (logMovement) Debug.Log($"[Movement] Lean right ({leanStepDegrees:F1} deg)");

        ApplyLeanRotation(leanStepDegrees);
        animator.SetTrigger("LeanRight");
    }

    private void ApplyLeanRotation(float yawDelta)
    {
        // Lean gestures should rotate the player body, not the camera.
        Transform target = transform;

        if (leanUseLocalRotation)
            target.Rotate(0f, yawDelta, 0f, Space.Self);
        else
            target.Rotate(0f, yawDelta, 0f, Space.World);
    }
    
    public void Jump()
    {
        if (!isGrounded || isSliding) return;
        
        if (logMovement) Debug.Log("[Movement] Jump");
        
        verticalVelocity = Mathf.Sqrt(jumpHeight * -2f * gravity);
        animator.SetTrigger("Jump");
    }
    
    public void Slide()
    {
        if (isSliding || !isRunning) return;
        
        if (logMovement) Debug.Log("[Movement] Slide");
        
        isSliding = true;
        slideEndTime = Time.time + slideDuration;
        
        animator.SetTrigger("Slide");
        animator.SetBool("isSliding", true);
        
        // Disable running flag
        isRunning = false;
        animator.SetBool("isRunning", false);
    }
    
    private void StopSlide()
    {
        if (!isSliding) return;
        
        if (logMovement) Debug.Log("[Movement] Stop slide");
        
        isSliding = false;
        animator.SetBool("isSliding", false);
        
        // Return to walking
        Walk();
    }
    
    public void SetSpeed(float speed)
    {
        currentSpeed = speed;
        animator.SetFloat("Speed", speed);
    }

    public void RefreshRunSignal()
    {
        runDrivenByGesture = true;
        lastRunSignalTime = Time.time;
    }
    
    #endregion
    
    #region Movement Logic
    
    private void MoveForward()
    {
        if (characterController == null || !characterController.enabled)
            return;
            
        Vector3 move = transform.forward * currentSpeed;
        move.y = verticalVelocity;
        
        characterController.Move(move * Time.deltaTime);
    }
    
    private void ApplyGravity()
    {
        if (characterController == null)
            return;
            
        if (isGrounded && verticalVelocity < 0)
        {
            verticalVelocity = -2f; // Small downward force to keep grounded
        }
        else
        {
            verticalVelocity += gravity * Time.deltaTime;
        }
        
        // Apply vertical movement
        Vector3 move = new Vector3(0, verticalVelocity, 0);
        characterController.Move(move * Time.deltaTime);
    }
    
    private void CheckGrounded()
    {
        if (characterController == null)
            return;
            
        // Simple grounded check
        isGrounded = characterController.isGrounded;
        animator.SetBool("isGrounded", isGrounded);
    }
    
    #endregion
    
    #region Public Properties
    
    public bool IsRunning()
    {
        return isRunning;
    }
    
    public bool IsWalking()
    {
        return isWalking;
    }
    
    public bool IsCrouching()
    {
        return isCrouching;
    }
    
    public bool IsSliding()
    {
        return isSliding;
    }
    
    public bool IsMoving()
    {
        return isRunning || isWalking || isSliding;
    }
    
    #endregion
}
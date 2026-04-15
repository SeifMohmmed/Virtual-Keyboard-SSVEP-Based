/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2026 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
/*
 * main.c
 *
 * This code measures the output from a bioamplifier circuit
 * and sends the values over to a PC over USART.
 * On startup, the code waits for the PC to tell it to start.
 *
 * Created By STM32CubeMX
 *
 * Fix 1: Replaced broken RXNE interrupt workaround with simple
 *         HAL_UART_Receive polling for the startup handshake.
 *         The original HAL internal pointer hack was not reliably
 *         triggering HAL_UART_RxCpltCallback on STM32F303.
 * Fix 2: HAL_printf("starting\n") - added \n so Python readline()
 *         detects end of handshake string.
 * Fix 3: strncmp uses rx_control.buffer + rx_control.tail directly
 *         instead of print_str_ptr to avoid NULL pointer issues.
 */

#include <string.h>
#include "utils.h"

/* Enable Printing of Receive Messages From USART */
#define RECEIVE_PRINT_EN 0
#define ADC_BUFF_SIZE 512
#define ADC_BUFF_HALF_SIZE ADC_BUFF_SIZE / 2
/* The offset is the size of adc array as this accounts for casting 16bit to 8bit int */
#define ADC_BUFF_OFFSET ADC_BUFF_SIZE
#define UART_RX_BUFFER_SIZE 256
#define END_OF_MESSAGE_CHAR '\n'
#define START_STR "start\n"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;
DMA_HandleTypeDef hdma_adc1;

TIM_HandleTypeDef htim3;

UART_HandleTypeDef huart2;
DMA_HandleTypeDef hdma_usart2_tx;

/* USER CODE BEGIN PV */

/* __IO is volatile in HAL library */
__IO uint8_t *print_str_ptr = NULL;
__IO uint16_t adc_val[ADC_BUFF_SIZE] = {0}; /* Where the ADC values are stored */
__IO uint16_t adc_array_offset = 0;         /* Offset for sending first or second half of ADC buffer */
__IO uint8_t adc_print = 0;                 /* Print adc values flag */
__IO uint8_t str_print = 0;                 /* Print string flag */
__IO uint8_t end_of_message = 0;            /* Flag is set when start of string has been sent */
__IO uint8_t start = 0;                     /* Flag to start main loop after receiving response from PC */

uint8_t rx_buffer[UART_RX_BUFFER_SIZE] = {0}; /* Buffer for received messages over USART */
circle_buff_s rx_control = {                  /* Struct for managing receive buffer */
                            .buffer = rx_buffer,
                            .head = 0,
                            .tail = 0,
                            .size = UART_RX_BUFFER_SIZE,
                            .wrapped = 0};

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_ADC1_Init(void);
static void MX_TIM3_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  MX_GPIO_Init();
  MX_DMA_Init();
  MX_ADC1_Init();
  MX_TIM3_Init();
  MX_USART2_UART_Init();

  /* USER CODE BEGIN 2 */

  /* -------------------------------------------------------------------------
   * FIX 1: STARTUP HANDSHAKE using HAL_UART_Receive polling
   *
   * The original code used a fragile hack (manually setting huart2.pRxBuffPtr)
   * to try to receive bytes via interrupt before the HAL was fully set up.
   * This did not work reliably on STM32F303.
   *
   * Instead we now poll for bytes with a 1ms timeout inside a loop that also
   * sends "s" every 500ms. When a full line ending with '\n' is received,
   * we check if it equals "start\n" and proceed if it does.
   * ------------------------------------------------------------------------- */
  {
    uint8_t handshake_buf[16] = {0};
    uint8_t handshake_idx = 0;
    uint8_t rx_byte = 0;
    uint32_t last_send_tick = 0;

    while (!start)
    {
      /* Send "s" every 500ms */
      if ((HAL_GetTick() - last_send_tick) >= 500)
      {
        HAL_printf("s");
        last_send_tick = HAL_GetTick();
      }

      /* Poll for one byte with 1ms timeout (non-blocking feel) */
      if (HAL_UART_Receive(&huart2, &rx_byte, 1, 1) == HAL_OK)
      {
        /* Store byte, guard against overflow */
        if (handshake_idx < sizeof(handshake_buf) - 1)
        {
          handshake_buf[handshake_idx++] = rx_byte;
        }
        else
        {
          /* Overflow - reset buffer */
          memset(handshake_buf, 0, sizeof(handshake_buf));
          handshake_idx = 0;
          handshake_buf[handshake_idx++] = rx_byte;
        }

        /* End of message received */
        if (rx_byte == END_OF_MESSAGE_CHAR)
        {
          /* Check for "start\n" */
          if (strncmp((char *)handshake_buf, START_STR, strlen(START_STR)) == 0)
          {
            start = 1;
          }
          /* Reset for next message */
          memset(handshake_buf, 0, sizeof(handshake_buf));
          handshake_idx = 0;
        }
      }
    }
  }

  /* FIX 2: Added \n so Python readline() detects end of this response */
  HAL_printf("starting\n");

  /* Wait for UART DMA to finish sending "starting\n" before touching ADC.
   * Without this, DMA is still busy when ADC calibration starts → HAL_BUSY
   * → Error_Handler → red LED. */
  HAL_Delay(200);

  /* Calibrate ADC */
  HAL_Status_Check(HAL_ADCEx_Calibration_Start(&hadc1, ADC_SINGLE_ENDED));

  /* Start DMA controller taking samples from ADC and placing them in buffer */
  HAL_Status_Check(HAL_ADC_Start_DMA(&hadc1, (uint32_t *)&adc_val, ADC_BUFF_SIZE));

  /* Start timer used to trigger ADC conversions, timer set to 1KHz */
  HAL_Status_Check(HAL_TIM_Base_Start_IT(&htim3));

  /* Enable interrupt for runtime UART receive (after handshake is complete) */
  __HAL_UART_ENABLE_IT(&huart2, UART_IT_RXNE);
  huart2.pRxBuffPtr = rx_control.buffer;
  huart2.Mask = 0xFFU;

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* ADC values ready to send */
    if (adc_print)
    {
      /* Sending half the ADC buffer: 512 uint8_t's = 256 uint16_t's */
      HAL_print_raw((uint8_t *)(&adc_val) + adc_array_offset, ADC_BUFF_SIZE);
      adc_print = 0;
    }

    /* Received message over USART, print it back (only if RECEIVE_PRINT_EN=1) */
    if (str_print && RECEIVE_PRINT_EN)
    {
      if (!end_of_message)
      {
        HAL_printf("Received %s", print_str_ptr);
      }
      else
      {
        HAL_print_raw((uint8_t *)print_str_ptr, rx_control.head);
        HAL_print_raw((uint8_t *)'\n', 1);
        end_of_message = 0;
      }
      str_print = 0;
    }
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL16;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
 * @brief ADC1 Initialization Function
 * @param None
 * @retval None
 */
static void MX_ADC1_Init(void)
{
  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_MultiModeTypeDef multimode = {0};
  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */

  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV1;
  hadc1.Init.Resolution = ADC_RESOLUTION_12B;
  hadc1.Init.ScanConvMode = ADC_SCAN_DISABLE;
  hadc1.Init.ContinuousConvMode = DISABLE;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_RISING;
  hadc1.Init.ExternalTrigConv = ADC_EXTERNALTRIGCONV_T3_TRGO;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion = 1;
  hadc1.Init.DMAContinuousRequests = ENABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  hadc1.Init.LowPowerAutoWait = DISABLE;
  hadc1.Init.Overrun = ADC_OVR_DATA_OVERWRITTEN;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  multimode.Mode = ADC_MODE_INDEPENDENT;
  if (HAL_ADCEx_MultiModeConfigChannel(&hadc1, &multimode) != HAL_OK)
  {
    Error_Handler();
  }

  sConfig.Channel = ADC_CHANNEL_1;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SingleDiff = ADC_SINGLE_ENDED;
  sConfig.SamplingTime = ADC_SAMPLETIME_1CYCLE_5;
  sConfig.OffsetNumber = ADC_OFFSET_NONE;
  sConfig.Offset = 0;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */
}

/**
 * @brief TIM3 Initialization Function
 * @param None
 * @retval None
 */
static void MX_TIM3_Init(void)
{
  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */

  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 100 - 1;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 640;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }

  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim3, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }

  sMasterConfig.MasterOutputTrigger = TIM_TRGO_UPDATE;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */
}

/**
 * @brief USART2 Initialization Function
 * @param None
 * @retval None
 */
static void MX_USART2_UART_Init(void)
{
  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */

  huart2.Instance = USART2;
  huart2.Init.BaudRate = 57600;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  huart2.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart2.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }

  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */
}

/**
 * Enable DMA controller clock
 */
static void MX_DMA_Init(void)
{
  __HAL_RCC_DMA1_CLK_ENABLE();

  HAL_NVIC_SetPriority(DMA1_Channel1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel1_IRQn);

  HAL_NVIC_SetPriority(DMA1_Channel7_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel7_IRQn);
}

/**
 * @brief GPIO Initialization Function
 * @param None
 * @retval None
 */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  __HAL_RCC_GPIOF_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_RESET);

  GPIO_InitStruct.Pin = GPIO_PIN_1;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOF, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = GPIO_PIN_1 | GPIO_PIN_3 | GPIO_PIN_4 | GPIO_PIN_5 | GPIO_PIN_6 | GPIO_PIN_7 | GPIO_PIN_8 | GPIO_PIN_9 | GPIO_PIN_10 | GPIO_PIN_11 | GPIO_PIN_12;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = GPIO_PIN_0 | GPIO_PIN_1 | GPIO_PIN_4 | GPIO_PIN_5 | GPIO_PIN_6 | GPIO_PIN_7;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = LED_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LED_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* Called when the ADC has filled half of the buffer */
void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc)
{
  adc_print = 1;
  adc_array_offset = 0;
}

/* Called when the ADC has filled the buffer */
void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc)
{
  adc_print = 1;
  adc_array_offset = ADC_BUFF_OFFSET;
  HAL_Status_Check(HAL_ADC_Start_DMA(&hadc1, (uint32_t *)&adc_val, ADC_BUFF_SIZE));
}

/* Called when UART receives 1 byte (runtime use only, after handshake) */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  /* End of buffer - wrap around */
  if ((uint32_t)huart->pRxBuffPtr >= ((uint32_t)(rx_control.buffer) + rx_control.size - 1))
  {
    print_str_ptr = rx_control.buffer + rx_control.tail;
    huart->pRxBuffPtr = rx_control.buffer;
    rx_control.head = 0;
    rx_control.tail = 0;
    rx_control.wrapped = 1;
    str_print = 1;
  }
  else
  {
    rx_control.head++;
  }

  /* Full message received */
  if ((uint8_t)*(rx_control.buffer + rx_control.head - 1) == (uint8_t)END_OF_MESSAGE_CHAR)
  {
    if (!str_print)
    {
      print_str_ptr = rx_control.buffer + rx_control.tail;
      if (RECEIVE_PRINT_EN)
      {
        str_print = 1;
      }
    }

    /* FIX 3: Compare directly from buffer+tail, not from print_str_ptr */
    if (!start && !(strncmp((char *)(rx_control.buffer + rx_control.tail), START_STR, strlen(START_STR))))
    {
      start = 1;
    }

    rx_control.tail = rx_control.head;

    if (rx_control.wrapped)
    {
      rx_control.wrapped = 0;
      memset(rx_control.buffer + rx_control.tail, 0, rx_control.size - rx_control.tail);
      end_of_message = 1;
    }
  }

  /* Re-enable byte received interrupt */
  __HAL_UART_ENABLE_IT(huart, UART_IT_RXNE);
}

/* USER CODE END 4 */

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

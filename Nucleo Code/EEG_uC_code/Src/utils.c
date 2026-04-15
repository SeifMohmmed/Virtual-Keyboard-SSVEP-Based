/*
 * utils.c
 *    Various user created functions
 *
 *  Fix 1: HAL_Status_Check now tolerates HAL_BUSY (returns without crashing)
 *          instead of calling Error_Handler, which caused red LED when ADC
 *          calibration ran while UART DMA was still busy.
 *  Fix 2: HAL_print_raw and HAL_printf_valist now have a timeout on the
 *          UART state wait loop to prevent infinite blocking.
 */

#include "utils.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#define PRINT_BUFFER_SIZE 256
#define UART_TIMEOUT_MS 100 /* Max ms to wait for UART to be ready */

extern UART_HandleTypeDef huart2;

/* Handles errors from HAL functions.
 * HAL_BUSY is tolerated — it means a previous DMA transfer is still running.
 * Only hard errors (HAL_ERROR, HAL_TIMEOUT) trigger Error_Handler. */
void HAL_Status_Check(HAL_StatusTypeDef status)
{
    if (status == HAL_ERROR || status == HAL_TIMEOUT)
    {
        Error_Handler();
    }
    /* HAL_BUSY and HAL_OK are both acceptable */
}

/* Wait for UART to be ready with a timeout to avoid infinite blocking */
static void wait_uart_ready(void)
{
    uint32_t start = HAL_GetTick();
    while (HAL_UART_GetState(&huart2) != HAL_UART_STATE_READY)
    {
        if ((HAL_GetTick() - start) > UART_TIMEOUT_MS)
        {
            /* Timeout — abort current transfer and try to recover */
            HAL_UART_Abort(&huart2);
            HAL_Delay(10);
            break;
        }
    }
}

/* Prints raw bytes over UART DMA */
void HAL_print_raw(uint8_t *byte, uint16_t size)
{
    wait_uart_ready();
    HAL_Status_Check(HAL_UART_Transmit_DMA(&huart2, byte, size));
}

/* Format string from args and print over USART.
 * Note: prepends 's' to distinguish string data from raw ADC binary data. */
void HAL_printf_valist(const char *fmt, va_list argp)
{
    char string[PRINT_BUFFER_SIZE] = {0};

    /* Prepend 's' to differentiate string print from raw ADC data */
    string[0] = 's';

    wait_uart_ready();

    if (vsnprintf(string + 1, PRINT_BUFFER_SIZE - 2, fmt, argp) > 0)
    {
        /* Append '\n' if not already present */
        size_t len = strlen(string);
        if (string[len - 1] != '\n')
        {
            string[len] = '\n';
        }
        HAL_Status_Check(HAL_UART_Transmit_DMA(&huart2, (uint8_t *)string, strlen(string)));
    }
    else
    {
        HAL_Status_Check(HAL_UART_Transmit_DMA(&huart2, (uint8_t *)"E - Print\n", 10));
    }
}

/* Format and send string over USART */
void HAL_printf(const char *fmt, ...)
{
    va_list argp;
    va_start(argp, fmt);
    HAL_printf_valist(fmt, argp);
    va_end(argp);
}

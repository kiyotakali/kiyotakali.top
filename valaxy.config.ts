import { defineConfig } from 'valaxy'
import type { ThemeConfig } from 'valaxy-theme-hairy'
import { addonWaline } from 'valaxy-addon-waline'
import { addonMeting } from 'valaxy-addon-meting'

/**
 * User Config
 * do not use export const config to avoid defu conflict
 */
export default defineConfig<ThemeConfig>({
  theme: 'hairy',
  url: 'http://kiyotakali.top/',
  themeConfig:{
    theme: 'dark',
    
    nav: [
      {
        text: 'Home',
        link: '/',
        icon: 'i-material-symbols-home-work-sharp',
      },
      {
        text: 'About',
        link: '/about',
        icon: 'i-material-symbols-recent-actors-rounded',
      },
      {
        text: 'Posts',
        link: '/archives/',
        icon: 'i-material-symbols-import-contacts-rounded',
      },
      {
        text: 'Links',
        link: '/links/',
        icon: 'i-material-symbols-monitor-heart',
      },
      {
        text: 'CV',
        link: 'https://xjtu-wjz.github.io/',
        icon: 'i-ri-sd-card-mini-fill',
      },
      {
        text: 'Github',
        link: 'https://github.com/xjtu-wjz',
        icon: 'i-ri-github-fill',
      },
    ],

    footer: {
      since: 2024,
      beian: {
        enable: false,
        icp: '',
      },
      powered: true,
    },


  },




  addons: [
    addonMeting({
      global: false,
      props: {
        // 设置你的网易云/qq或其他歌单 ID
        id: '5312894314',
        type: 'playlist',
        autoplay: false,
        theme: 'var(--hy-c-primary)',
      },
    }),
    // 请参考 https://waline.js.org/ 设置 serverURL 地址
    addonWaline({
      comment: true,
      serverURL: '...',
      emoji: [/*  */],
      pageview: true,
    }),
  ],


})